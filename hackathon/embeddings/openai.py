from hackathon.embeddings import EmbeddingsBase
import openai
import os


class OpenAIEmbeddings(EmbeddingsBase):
    def __init__(self, api_key = None):
        super().__init__()
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
            assert isinstance(api_key, str)

        openai.api_key = api_key
        self.client = openai.Embedding()


    def get_embeddings(self, text):
        return self.client.create(
            model="text-embedding-ada-002",
            input=text
        )