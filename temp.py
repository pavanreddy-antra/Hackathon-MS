import openai
import os

from dotenv import load_dotenv
load_dotenv()

from hackathon.embeddings import AzureEmbeddings

from hackathon.chat import AzureChat
from hackathon.stt import AzureSTT

# openai.api_key = os.getenv("AZURE_OPENAI_KEY")
# openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
# openai.api_type = 'azure'
# openai.api_version = '2024-02-01'
# deployment_name = os.getenv("DEPLOYMENT_NAME")
#
# print(deployment_name)
#
# print(openai.Embedding.create(
#     deployment_id=deployment_name,
#     model="text-embedding-ada-002",
#     input="Hello"
# ))

stt = AzureSTT()

print(stt.get_text("audio.wav"))