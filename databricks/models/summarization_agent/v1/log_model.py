import mlflow
import os
import openai
from dotenv import load_dotenv
import pandas as pd
from mlflow.models.signature import infer_signature

# Ensure your OPENAI_API_KEY is set in your environment


class OpenAIWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load a specific version using URI syntax
        uc_schema = context.model_config["uc_schema"]
        prompt_name = context.model_config["prompt_name"]
        prompt_version = context.model_config["prompt_version"]
        num_sentences = context.model_config.get("num_sentences", 20)

        if not isinstance(num_sentences, int) or num_sentences <= 0:
            raise ValueError("Number of sentences must be a positive integer.")

        self.num_sentences = num_sentences
        self.prompt = mlflow.genai.load_prompt(
            name_or_uri=f"prompts:/{uc_schema}.{prompt_name}/{prompt_version}"
        )

    def format_inputs(self, model_input):
        content = '\n'.join(list(model_input["content"]))
        print(f"Content to summarize: {content}")
        # insert some code that formats your inputs
        # print(type(model_input))
        # model_input = model_input.to_dict()
        # content = model_input.get("content", "")
        if not content:
            raise ValueError("Content must be provided for summarization.")

        self.formatted_prompt = self.prompt.format(
            content=content, num_sentences=self.num_sentences
        )
        print(f"Formatted prompt: {self.formatted_prompt}")
        return self.formatted_prompt

    def format_outputs(self, outputs):
        if isinstance(outputs, list):
            return outputs
        if isinstance(outputs, str):
            # If the output is a single string, return it as a list
            return [outputs]
        else:
            raise ValueError("Output must be a string or a list of strings.")
        

    def predict(self, context, model_input):
        input_content = self.format_inputs(model_input)
        # Create an OpenAI client connected to OpenAI SDKs
        client = openai.OpenAI()
        model_name = context.model_config.get("openai_model", "gpt-4o-mini")
        # Format with variables
        response = client.responses.create(
            model=model_name,  # This example uses a Databricks hosted LLM - you can replace this with any AI Gateway or Model Serving endpoint. If you provide your own OpenAI credentials, replace with a valid OpenAI model e.g., gpt-4o, etc.
            input=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": input_content,
                },
            ],
        )
        print(response.output_text)
        # print(type(response.output_text))
        return self.format_outputs(response.output_text)


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env.local")
    
    # --- define input example as a DataFrame ---
    input_example_df = pd.DataFrame({"content": ["This is a sample text to summarize.", "see what happens when breaking it into multiple lines."]})

    # --- create a MOCK output to infer signature (list[str]) ---
    mock_output = ["...summary..."]  # same length/type as what predict() would return
    
    signature = infer_signature(input_example_df, mock_output)

    mlflow.pyfunc.log_model(
        name="v1_prompt_model",
        python_model=OpenAIWrapper(),
        input_example=input_example_df,
        registered_model_name="workspace.summarization_agent.v1_prompt_model",
        model_config={
            "openai_model": "gpt-4o-mini",
            "openai_key_env": os.environ["OPENAI_API_KEY"],
            "uc_schema": "workspace.summarization_agent",
            "prompt_name": "summarization_prompt",
            "prompt_version": "1",
            "num_sentences": 20,
        },
        signature=signature,
    )
