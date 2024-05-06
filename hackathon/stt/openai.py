from hackathon.stt import STTBase
import openai
from openai import OpenAI
import os

from dotenv import load_dotenv
load_dotenv()

class OpenAISTT(STTBase):
    def __init__(self, api_key):
        super().__init__()
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
            assert isinstance(api_key, str)

        openai.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def get_text(self, audio_data):
        with open(audio_data, 'rb') as audio_file:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return response['text']