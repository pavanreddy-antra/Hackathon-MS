from hackathon.chat import ChatBase
import google.generativeai as genai
import os

class GeminiChat(ChatBase):
    def __init__(self, api_key=None):
        super().__init__()
        if api_key is None:
            api_key = os.getenv('GEMINI_KEY')
            assert isinstance(api_key, str), "API key must be a string"

        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 8192,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
        ]

        # self.model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
        #                               generation_config=generation_config,
        #                               safety_settings=safety_settings)

        self.model = genai.GenerativeModel('gemini-pro',
                                            generation_config=generation_config,
                                            safety_settings=safety_settings)
        

    def get_response(self, prompt, history=None):
        if not isinstance(prompt, str):
            raise TypeError(f"Invalid Type for Messages: {type(prompt)}")

        if history is None:
            history = []

        chat = self.model.start_chat(history=history)

        response = chat.send_message(prompt)

        return response.text, chat.history.copy()

