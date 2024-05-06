from abc import ABC, abstractmethod

class ChatBase(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_response(self, text, *args, **kwargs):
        pass