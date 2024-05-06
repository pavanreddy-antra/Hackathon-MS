from hackathon.embeddings import EmbeddingsBase
import os

from dotenv import load_dotenv
load_dotenv()

from openai import AzureOpenAI

class AzureEmbeddings(EmbeddingsBase):
    def __init__(self):
        super().__init__()
        self.client = AzureOpenAI(
            azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
            api_version='2024-02-01',
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_ad_token_provider=None
        )

    def get_embeddings(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return [item.embedding for item in response.data][0]
