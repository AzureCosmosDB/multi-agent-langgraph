import json
import os

import langchain_openai
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

aoai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-09-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
azure_openai_api_version = "2023-05-15"
azure_deployment_name = "gpt-4o"
model = AzureChatOpenAI(
    azure_deployment=azure_deployment_name,
    api_version=azure_openai_api_version,
    temperature=0,
)

def generate_embedding(text):
    response = aoai_client.embeddings.create(input=text, model=os.getenv("AZURE_OPENAI_EMBEDDINGDEPLOYMENTID"))
    json_response = response.model_dump_json(indent=2)
    parsed_response = json.loads(json_response)
    return parsed_response['data'][0]['embedding']