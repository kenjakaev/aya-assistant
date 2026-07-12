from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Aya Assistant API")


class UserMessage(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"status": "Aya is alive", "version": "0.1.0"}


@app.post("/chat")
def chat_endpoint(message: UserMessage):
    user_text = message.text
    return {"response": f"Hello, I've read {user_text}"}
