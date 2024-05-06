from abc import ABC, abstractmethod


class CSPBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def index_data(self, index_name, file_path, *args, **kwargs):
        pass

    @abstractmethod
    async def simple_hs(self, prompt, index_name, *args, **kwargs):
        pass

    @abstractmethod
    async def start_conversation(self, *args, **kwargs):
        pass

    @abstractmethod
    async def speech_to_text(self, audio_data, *args, **kwargs):
        pass
