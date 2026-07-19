from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from fastapi.responses import StreamingResponse
from threading import Thread
from pydantic import BaseModel
import torch

model_id = "google/gemma-2-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="auto"
)

chat_history = []

app = FastAPI(title="Aya Assistant API")
app.mount("/static", StaticFiles(directory="static"), name="static")


class UserMessage(BaseModel):
    text: str


@app.get("/")
def read_root():
    return FileResponse("static/index.html")


@app.post("/chat")
def chat_endpoint(message: UserMessage):
    try:
        chat_history.append({"role": "user", "content": message.text})

        while len(chat_history) > 20:
            chat_history.pop(0)
            chat_history.pop(0)

        chat_prompt = tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )

        with torch.no_grad():
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

        chat_history.append({"role": "model", "content": message.text})

        def text_streamer():
            for new_text in streamer:
                yield new_text

        return StreamingResponse(text_streamer(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=45738)
