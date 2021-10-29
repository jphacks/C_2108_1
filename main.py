from fastapi import FastAPI
from pydantic import BaseModel
from src.dialogue_model import make_reply
from starlette.middleware.cors import CORSMiddleware

class Memo(BaseModel):
    input_text: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/")
def reply_to_memo(memo: Memo):
    replies = make_reply(memo.input_text)
    return {"replies": replies, "input_text": memo.input_text}
