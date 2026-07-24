import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from fastapi.responses import StreamingResponse
from threading import Thread
from pydantic import BaseModel
import torch
from rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MainApp")

model_id = "google/gemma-2-2b-it"
tokenizer = None
model = None
rag = None
chat_history = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, rag

    logger.info("Launching the application: loading models and knowledge base...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(
        "cuda"
    )
    logger.info("The Gemma 2B model has been successfully loaded")

    rag = RAGEngine()
    try:
        knowledge_path = "data/knowledge.txt"
        with open(knowledge_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
            knowledge_base = [
                chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()
            ]
        rag.add_documents(knowledge_base)
        logger.info(
            f"Knowledge base successfully loaded ({len(knowledge_base)}) chunks"
        )
    except FileNotFoundError:
        logger.warning(
            f"File {knowledge_path} not found! RAG started with an empty database"
        )

    logger.info("The server is fully ready for operation!")

    yield

    logger.info("Stopping server: cleaning up resources...")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    logger.info("VRAM has been cleared")


app = FastAPI(title="Aya Assistant API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class UserMessage(BaseModel):
    text: str


@app.get("/")
def read_root():
    return FileResponse("static/index.html")


@app.post("/chat")
def chat_endpoint(message: UserMessage):
    try:
        logger.info(f"A request has been received from the user: {message.text}")
        context_chunks = rag.search(message.text, top_k=2)

        if context_chunks:
            logger.info(f"RAG: Relevant chunks found: {len(context_chunks)}")
            context_str = "\n".join(context_chunks)
            current_user_content = (
                f"Knowledge base context:\n{context_str}\n\nUser query: {message.text}"
            )
        else:
            logger.info("RAG: Matching context not found (score below threshold)")
            current_user_content = f"User query: {message.text}"

        messages_for_model = []
        messages_for_model.extend(chat_history)
        messages_for_model.append({"role": "user", "content": current_user_content})

        chat_prompt = tokenizer.apply_chat_template(
            messages_for_model, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.45,
            top_p=0.85,
            do_sample=True,
        )

        with torch.no_grad():
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

        def text_streamer():
            full_response_chunks = []
            for new_text in streamer:
                full_response_chunks.append(new_text)
                yield new_text

            full_response = "".join(full_response_chunks)
            logger.info(
                "The response has been successfully generated and sent to the user."
            )

            chat_history.append({"role": "user", "content": message.text})
            chat_history.append({"role": "model", "content": full_response})

            while len(chat_history) > 20:
                chat_history.pop(0)
                chat_history.pop(0)

        return StreamingResponse(text_streamer(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=45738)
