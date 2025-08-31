import datetime
import hashlib
from clerk_backend_api import AuthenticateRequestOptions, Clerk
from databricks import sql
import openai
import requests
import json
from fastapi import HTTPException, Request
import os


def get_current_user_id(request: Request):
    with Clerk(bearer_auth=os.getenv("CLERK_SECRET_KEY")) as clerk:
        state = clerk.authenticate_request(
            request,
            AuthenticateRequestOptions(
                authorized_parties=[os.getenv("FRONTEND_ORIGIN"), 'http://localhost:3000'],
                # 👇 Accept the audience you set in the template:
                audience=["convex"],
            ),
        )

    if not state.is_signed_in:
        # state.reason explains why (expired, invalid signature, wrong aud, etc.)
        raise HTTPException(status_code=401, detail=str(state.reason))

    # The verified JWT payload is here:
    return state.payload.get("sub")


def get_ab_test_group(user_id: str) -> str:
    """
    Hashes a user ID and assigns it to an 'Even (Group A)' or 'Odd (Group B)' group.

    Args:
        user_id (str): The unique identifier for the user.

    Returns:
        str: The assigned A/B test group ('Even (Group A)' or 'Odd (Group B)').
    """
    # Use SHA-256 for a more robust and evenly distributed hash.
    # It's important to encode the string to bytes before hashing.
    hashed_id = hashlib.sha256(user_id.encode("utf-8")).hexdigest()

    # Convert the hexadecimal hash to an integer.
    # We only need a portion of the hash to ensure a wide distribution
    # while keeping the number manageable. Taking the first 8 characters
    # of the hex string provides enough entropy for this purpose.
    hash_as_int = int(hashed_id[:8], 16)

    # Determine if the integer hash is even or odd
    return hash_as_int % 2


def load_prompt(prompt_file):
    prompt = ""
    with open(prompt_file, "r", 1) as reader:
        prompt += reader.readline()

    return prompt


def direct_summary(input_content, user_id):
    exp_params = json.load(open("experiment.config.json"))
    prompt1_file = exp_params["exp_direct_summarization"]["prompt1"]
    prompt2_file = exp_params["exp_direct_summarization"]["prompt2"]

    if not prompt1_file or not prompt2_file:
        raise HTTPException(
            status_code=500, detail="prompt file not found in experiment.config.json"
        )

    prompt1 = load_prompt(prompt1_file)
    prompt2 = load_prompt(prompt2_file)

    model_name = exp_params["exp_direct_summarization"]["model_name"]
    if not model_name:
        raise HTTPException(
            status_code=500, detail="model_name not found in experiment.config.json"
        )

    assigned_group = get_ab_test_group(user_id)
    if assigned_group == 0:
        prompt = prompt1
    else:
        prompt = prompt2

    # Create an OpenAI client connected to OpenAI SDKs
    client = openai.OpenAI()
    from openai import OpenAI

    client = OpenAI()

    # print(f"prompt:{prompt}")
    # Format with variables
    response = client.responses.create(
        model=model_name,  # This example uses a Databricks hosted LLM - you can replace this with any AI Gateway or Model Serving endpoint. If you provide your own OpenAI credentials, replace with a valid OpenAI model e.g., gpt-4o, etc.
        input=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": input_content,
            },
        ],
    )
    # print(f"output:{response.output_text}")
    return response.output_text


def score_model(dataset, user_id, client_request_id):
    exp_params = json.load(open("experiment.config.json"))
    endpoint1_url = exp_params["exp_summarization"]["endpoint1"]
    endpoint2_url = exp_params["exp_summarization"]["endpoint2"]

    if not endpoint1_url or not endpoint2_url:
        raise HTTPException(
            status_code=500, detail="Endpoint URL not found in experiment.config.json"
        )

    assigned_group = get_ab_test_group(user_id)
    if assigned_group == 0:
        url = endpoint1_url
    else:
        url = endpoint2_url

    headers = {
        "Authorization": f'Bearer {os.environ.get("DATABRICKS_TOKEN")}',
        "Content-Type": "application/json",
    }

    ds_dict = {
        "client_request_id": client_request_id,
        "dataframe_split": dataset.to_dict(orient="split"),
    }

    data_json = json.dumps(ds_dict, allow_nan=True)
    if not data_json:
        raise HTTPException(
            status_code=500, detail="Data must be provided for scoring."
        )

    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    # print(f"Response : {response}")
    if not response or response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    response_json = response.json()
    # Example response_json:
    # {"predictions": ["The summarization endpoint is designed to process tasks involving content summarization. \n It accepts input in the form of text, which can be any length of content that needs to be condensed."], "databricks_output": {"trace": null, "databricks_request_id": "fe6a397d-a7a3-4fca-aa18-51528cb3c009", "client_request_id": "abc123"}
    if not "predictions" in response_json:
        raise HTTPException(
            status_code=500, detail="Predictions not found in the response."
        )

    text = response_json["predictions"][0]

    return text


def submit_feedback(feedback: dict, user_id: str):
    """Submits user feedback to a Databricks SQL Warehouse."""
    try:
        with sql.connect(
            server_hostname=os.getenv("DATABRICKS_HOST"),
            http_path=os.getenv("DATABRICKS_WAREHOUSE_HTTP_PATH"),
            access_token=os.getenv("DATABRICKS_TOKEN"),
        ) as connection:
            with connection.cursor() as cursor:
                # print("Successfully connected to Databricks SQL Warehouse.")
                cursor.execute(
                    """INSERT INTO workspace.summarization_agent.feedback
            (client_request_id, user_id, feedback_rate, feedback_text, feedback_timestamp)
            VALUES (?, ?, ?, ?, ?)""",
                    (
                        feedback["client_request_id"],
                        user_id,
                        feedback["rate"],
                        feedback["comment"],
                        datetime.datetime.utcnow(),
                    ),
                )

                # print(f"Insert info succeeded.")
    except Exception as e:
        print(f"Error connecting to Databricks SQL Warehouse: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Example usage with your user ID
    user_id_example = "abc123"
    assigned_group = get_ab_test_group(user_id_example)

    print(f"User ID: {user_id_example}")
    print(f"Assigned A/B Test Group: {assigned_group}")
