import os
import mlflow
import argparse
from dotenv import load_dotenv

"""This script is used to register a prompt in Unity Catalog in Databricks.
    It requires the prompt name, Unity Catalog schema, and the path to the prompt file.
    The prompt file should contain the prompt template in a format compatible with MLflow GenAI.
    Make sure you have the necessary permissions to create functions and manage prompts in the specified schema.
    The script will register the prompt and print the details of the created prompt.
    Ensure you have the MLflow library installed and configured to connect to your Databricks workspace.
    The script will read the prompt template from the specified file and register it in Unity Catalog
    under the specified schema. The prompt will be registered with a commit message and tags for better organization."""
    

load_dotenv(dotenv_path=".env.local")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit a prompt to Unity Catalog in Databricks",
        usage="python3 submit_prompt.py --prompt_name summarization_prompt --uc_schema workspace.default --prompt_file databricks/prompts/v1_prompt.txt",
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        required=True,
        help="Name of the prompt to be registered",
    )
    parser.add_argument(
        "--uc_schema",
        type=str,
        required=True,
        help="Unity Catalog schema where the prompt will be registered",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path to the prompt file to be registered",
    )

    args = parser.parse_args()
    print(args.prompt_name, args.uc_schema, args.prompt_file)

    initial_template = ""
    with open(args.prompt_file, "r") as file:
        for line in file:
            if line.startswith("#"):
                # Skip comment lines
                continue
            initial_template += line.strip() + "\n"

    print(f"Initial template read from file: {initial_template}")

    # Register a new prompt
    prompt = mlflow.genai.register_prompt(
        name=f"{args.uc_schema}.{args.prompt_name}",
        template=initial_template,
        # all parameters below are optional
        commit_message=f"{args.prompt_name} prompt registered",
        tags={
            "author": "data-science-team@company.com",
            "use_case": "document_summarization",
            "task": "summarization",
            "language": "en",
            "model_compatibility": "gpt-4o-mini",
        },
    )
    
    print(f"Created prompt '{prompt.name}' (version {prompt.version})")
