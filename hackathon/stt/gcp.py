from hackathon.stt import STTBase
from google.cloud import speech
from google.oauth2 import service_account
import os


class GCPSTT(STTBase):
    def __init__(self, api_key=None):
        super().__init__()
        if api_key is None:
            api_key = os.getenv('GOOGLE_STT_KEY')
            print(api_key)
        
        credentials = service_account.Credentials.from_service_account_file(os.getenv("GCP_CREDENTIALS_PATH"))
        self.client = speech.SpeechClient(credentials=credentials)

    async def get_text(self, audio_data):
        with open(audio_data, "rb") as audio_file:
            audio_content = audio_file.read()

        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )

        response = self.client.recognize(config=config, audio=audio)
        
        for result in response.results:
            return result.alternatives[0].transcript