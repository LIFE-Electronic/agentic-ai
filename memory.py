from abc import ABC, abstractmethod
from llm import LLMMessage

class MemoryProvider(ABC):
    @abstractmethod
    async def store(self, message: LLMMessage) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def retrieve_many(self, limit: int | None) -> list[LLMMessage]:
        raise NotImplementedError()
    
    @abstractmethod
    async def clear(self) -> None:
        raise NotImplementedError()

class RAMMemory(MemoryProvider):

    _memory_bank: list[LLMMessage]

    def __init__(self):
        super().__init__()
        self._memory_bank = []

    async def store(self, message: LLMMessage) -> None:
        self._memory_bank.append(message)

    async def clear(self) -> None:
        self._memory_bank = []

    async def retrieve_many(self, limit: int | None = None) -> list[LLMMessage]:
        return self._memory_bank[:limit]