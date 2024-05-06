from hackathon.chat import ChatBase
import os

from dotenv import load_dotenv

load_dotenv()

from openai import AzureOpenAI


class AzureChat(ChatBase):
    def __init__(self, api_key=None):
        super().__init__()
        self.client = AzureOpenAI(
            azure_deployment=os.getenv("MODEL_DEPLOYMENT"),
            api_version='2024-02-01',
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_ad_token_provider=None
        )

    def get_response(self, message, history=None, system_prompt=None):
        if history is None:
            history = []

        history_copy = history.copy()

        if system_prompt is not None:
            history_copy.append({"role": "system", "content": system_prompt})

        if not isinstance(message, str):
            raise TypeError(f"Invalid Type {type(message)} for prompt")

        history.append({"role": "user", "content": message})
        history_copy.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=history_copy,
            temperature=0.5
        )

        response = [item.message.content for item in response.choices][0]
        history.append({"role": "assistant", "content": response})

        print(response)

        return response, history
