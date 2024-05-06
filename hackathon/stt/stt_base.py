from abc import ABC, abstractmethod


class STTBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def get_text(self):
        pass