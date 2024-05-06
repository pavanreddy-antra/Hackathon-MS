from hackathon.chat import ChatBase
import openai
import os


class OpenAIChat(ChatBase):
    def __init__(self, api_key=None):
        super().__init__()
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
            assert isinstance(api_key, str), "API key must be a string"

        openai.api_key = api_key
        self.client = openai.ChatCompletion()

    def get_response(self, messages):
        if isinstance(messages, str):
            messages = {"role": "user", "content": messages}

        if not isinstance(messages, list):
            if isinstance(messages, dict): 
                messages = [messages]
            else:
                raise TypeError(f"Invalid Type for Messages: {type(messages)}")
            
        return self.client.create(
            model="gpt-4",
            messages=messages
        )
    
