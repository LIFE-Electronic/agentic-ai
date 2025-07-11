from abc import ABC, abstractmethod
from llm import LLMMessage

class ShortTermMemoryProvider(ABC):
    @abstractmethod
    async def store(self, message: LLMMessage) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def retrieve_many(self, limit: int | None) -> LLMMessage:
        raise NotImplementedError()
    
    @abstractmethod
    async def clear(self) -> None:
        raise NotImplementedError()

class QueueBasedMemory(ShortTermMemoryProvider):

    _memory_bank = list[LLMMessage]

    def __init__(self):
        super().__init__()
        self._memory_bank = []

    async def store(self, message: LLMMessage) -> None:
        self._memory_bank.append(message)

    async def clear(self) -> None:
        self._memory_bank = []

    async def retrieve_many(self, limit: int | None = None) -> LLMMessage:
        return self._memory_bank[:limit]