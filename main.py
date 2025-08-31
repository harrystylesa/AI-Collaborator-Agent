from fastapi import FastAPI, HTTPException, Depends
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from tools import direct_summary, score_model, submit_feedback, get_current_user_id
from fastapi.middleware.cors import CORSMiddleware
import os

# Check if the .env.local file exists
if os.path.exists(".env.fastapi"):
    load_dotenv(dotenv_path=".env.fastapi")


class Summary(BaseModel):
    client_request_id: str
    content: str


class Feedback(BaseModel):
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

# This is the crucial part that handles the OPTIONS request
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        os.getenv("FRONTEND_ORIGIN"),
        os.getenv("FRONTEND_ORIGIN_DEV"),
    ],  # Replace with your Next.js frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/summarization/feedback")
async def root(feedback: Feedback, user_id: str = Depends(get_current_user_id)):
    """Feedback endpoint that accepts user feedback on the summarization."""
    if not feedback:
        raise HTTPException(status_code=400, detail="missing body")
    if not feedback.client_request_id:
        raise HTTPException(status_code=400, detail="missing client_request_id")
    if not feedback.rate or feedback.rate < 1 or feedback.rate > 5:
        raise HTTPException(status_code=400, detail="rate must be between 1 and 5")
    # Here you can implement logic to store feedback in a database or send it to Databricks
    # print(user_id)
    # import time

    # time.sleep(5)
    # # raise HTTPException(status_code=400, detail="rate must be between 1 and 5")

    # return feedbackResponse(client_request_id=feedback.client_request_id, status=200)

    feedback_data = feedback.dict()
    submit_feedback(feedback_data, user_id)
    return feedbackResponse(client_request_id=feedback.client_request_id, status=200)


@app.post("/summarization")
async def root(summary: Summary, user_id: str = Depends(get_current_user_id)):
    """Summarization endpoint that accepts a task and content, formats the input, and returns a summary."""
    if not summary:
        raise HTTPException(status_code=400, detail="missing body")
    if not summary.client_request_id:
        raise HTTPException(status_code=400, detail="missing client_request_id")
    if not summary.content:
        raise HTTPException(status_code=400, detail="missing content")

    contents = summary.content.split("\n")
    df = pd.DataFrame({"content": contents})
    # print(user_id)
    # import time
    # time.sleep(10)
    # return summaryResponse(
    #     client_request_id=summary.client_request_id, summary="this is a test summary"
    # )

    prediction = score_model(df, user_id, summary.client_request_id)
    return summaryResponse(
        client_request_id=summary.client_request_id, summary=prediction
    )


@app.post("/summarization_direct")
async def root(summary: Summary, user_id: str = Depends(get_current_user_id)):
    """Summarization endpoint that accepts a task and content, formats the input, and returns a summary."""
    if not summary:
        raise HTTPException(status_code=400, detail="missing body")
    if not summary.client_request_id:
        raise HTTPException(status_code=400, detail="missing client_request_id")
    if not summary.content:
        raise HTTPException(status_code=400, detail="missing content")

    content = summary.content.strip()

    prediction = direct_summary(content, user_id)
    return summaryResponse(
        client_request_id=summary.client_request_id, summary=prediction
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80, log_level=0)
