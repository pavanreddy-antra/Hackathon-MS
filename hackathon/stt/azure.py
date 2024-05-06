from hackathon.stt import STTBase
import os
import azure.cognitiveservices.speech as speechsdk

from dotenv import load_dotenv
load_dotenv()

class AzureSTT(STTBase):
    def __init__(self, api_key=None):
        super().__init__()
        speech_key = os.getenv('AZURE_SPEECH_KEY')

        service_region = "eastus"
        self.speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)



    def get_text(self, audio_data):
        audio_config = speechsdk.audio.AudioConfig(filename=audio_data)
        recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return "No speech could be recognized."
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            return f"Speech Recognition canceled: {cancellation_details}"
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                return f"Error details: {cancellation_details.error_details}"