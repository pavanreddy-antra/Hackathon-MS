from hackathon.embeddings import EmbeddingsBase
from google.cloud import aiplatform
from google.oauth2 import service_account
from vertexai.preview.language_models import TextEmbeddingModel
import os
import numpy as np


class GCPEmbeddings(EmbeddingsBase):
    def __init__(self, project_id=None, location='us-east1'):
        super().__init__()
        self.project_id = project_id or os.getenv('GCP_PROJECT_ID')
        self.location = location

        credentials = service_account.Credentials.from_service_account_file(os.getenv("GCP_CREDENTIALS_PATH"))
        aiplatform.init(credentials=credentials, project=self.project_id, location=self.location)

    def get_embeddings(self, text):
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko")
        embeddings = model.get_embeddings([text])
        return list(embeddings[0].values)

