import mlflow
import os
import openai
from dotenv import load_dotenv
import pandas as pd
from mlflow.models.signature import infer_signature

# Ensure your OPENAI_API_KEY is set in your environment


class OpenAIWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        promptfile = context.artifacts.get("prompt")
        prompt = ""
        with open(promptfile, 'r', 1) as reader:
            prompt += reader.readline()
            
        # print(f'loaded prompt: {prompt}')        
        self.prompt = prompt
        
    def format_inputs(self, model_input):
        content = '\n'.join(list(model_input["content"]))
        # print(f"Content to summarize: {content}")
        if not content:
            raise ValueError("Content must be provided for summarization.")

        # print(f"input content: {content}")
        return content

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
                    "content": self.prompt,
                },
                {
                    "role": "user",
                    "content": input_content,
                },
            ],
        )
        # print(response.output_text)
        # print(type(response.output_text))
        return self.format_outputs(response.output_text)


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env.model")
    
    # --- define input example as a DataFrame ---
    input_example_df = pd.DataFrame({"content": ["This is a sample text to summarize.", "see what happens when breaking it into multiple lines."]})

    # --- create a MOCK output to infer signature (list[str]) ---
    mock_output = ["...summary..."]  # same length/type as what predict() would return
    
    signature = infer_signature(input_example_df, mock_output)

    mlflow.pyfunc.log_model(
        name="summary_prompt_model",
        artifacts={"prompt": "databricks/models/summarization_agent/v2_prompts.txt"},
        python_model=OpenAIWrapper(),
        input_example=input_example_df,
        registered_model_name="workspace.summarization_agent.summary_prompt_model",
        model_config={
            "openai_model": "gpt-4o-mini",
            "openai_key_env": os.environ["OPENAI_API_KEY"]
        },
        signature=signature,
    )
