from abc import ABC, abstractmethod

class EmbeddingsBase(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_embeddings(self, text, *args, **kwargs):
        pass