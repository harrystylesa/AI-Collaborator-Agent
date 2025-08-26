from fastapi import FastAPI, HTTPException
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from tools import score_model, submit_feedback

load_dotenv(dotenv_path=".env.fastapi")

# 1 implement feedback api, register with client_request_id on databricks
# 2 implement client_request_id receive and return


class Summary(BaseModel):
    user_id: str
    client_request_id: str
    content: str


class Feedback(BaseModel):
    user_id: str
    client_request_id: str
    rate: int
    comment: str


class summaryResponse(BaseModel):
    client_request_id: str
    summary: str


class feedbackResponse(BaseModel):
    client_request_id: str
    status: int


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/summarization/feedback")
async def root(feedback: Feedback):
    """Feedback endpoint that accepts user feedback on the summarization."""
    if not feedback:
        raise HTTPException(status_code=400, detail="missing body")
    if not feedback.client_request_id:
        raise HTTPException(status_code=400, detail="missing client_request_id")
    if not feedback.user_id:
        raise HTTPException(status_code=400, detail="missing user_id")
    if not feedback.rate or feedback.rate < 1 or feedback.rate > 5:
        raise HTTPException(status_code=400, detail="rate must be between 1 and 5")
    # Here you can implement logic to store feedback in a database or send it to Databricks

    feedback_data = feedback.dict()
    submit_feedback(feedback_data)
    return feedbackResponse(client_request_id=feedback.client_request_id, status=200)


@app.post("/summarization")
async def root(summary: Summary):
    """Summarization endpoint that accepts a task and content, formats the input, and returns a summary."""
    if not summary:
        raise HTTPException(status_code=400, detail="missing body")
    if not summary.client_request_id:
        raise HTTPException(status_code=400, detail="missing client_request_id")
    if not summary.content:
        raise HTTPException(status_code=400, detail="missing content")
    if not summary.user_id:
        raise HTTPException(status_code=400, detail="missing user_id")

    contents = summary.content.split("\n")
    df = pd.DataFrame({"content": contents})
    prediction = score_model(df, summary.user_id, summary.client_request_id)

    return summaryResponse(client_request_id=summary.client_request_id, summary=prediction)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
