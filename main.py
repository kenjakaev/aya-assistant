from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
import torch

model_id = "google/gemma-2-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.float16, device_map="auto"
)

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
        messages = [{"role": "user", "content": message.text}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9
            )

        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=45738)
