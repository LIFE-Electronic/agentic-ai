from openai import DefaultAioHttpClient, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionDeveloperMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionToolParam
from openai.lib._parsing._completions import type_to_response_format_param
from openai._types import NotGiven
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from typing import Literal, Iterable, Callable, Any, Tuple, Awaitable
from abc import ABC, abstractmethod
from dataclasses import field, asdict
from httpx import AsyncClient
from util import prepare_prompt
import json

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OpenAIConfig():
    api_key: str
    model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    http_client: AsyncClient = DefaultAioHttpClient()

@dataclass
class LLMMessage:
    role: Literal['user', 'assistant', 'developer']
    content: str

type ToolDescription = dict[str, Any]

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ToolDefinition:
    description: ToolDescription
    function: Callable[..., Awaitable[str]]

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict[str, Any])

@dataclass
class ChatResponse:
    content: str = ''
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_log: list[LLMMessage] = field(default_factory=list)
    error: Literal['no_tool_found'] | None = None
    tokens_used: int = 0

@dataclass
class ModelResponse:
    name: str

@dataclass
class Embedding:
    embeddings: list[float]
    dims: int
    tokens_used: int = 0

class EmbeddingProvider(ABC):
    @abstractmethod
    async def create_embedding(self, text_content: str) -> Embedding:
        raise NotImplementedError()

class LLMProvider(ABC):
    @abstractmethod
    async def query(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition],
        force_tool_use: bool,
        response_format: Any | None,
    ) -> ChatResponse:
        raise NotImplementedError()

    @abstractmethod
    async def get_available_models(
        self,
    ) -> list[ModelResponse]:
        raise NotImplementedError()

class OpenAIProvider(LLMProvider, EmbeddingProvider):
    config: OpenAIConfig

    def __init__(self, config: OpenAIConfig):
        super().__init__()
        self.config = config

    async def create_embedding(self, text_content: str) -> Embedding:
        async with AsyncOpenAI(
            api_key=self.config.api_key,
            http_client=DefaultAioHttpClient(),
        ) as client:
            embedding_response = await client.embeddings.create(
                input=text_content,
                model=self.config.embedding_model,
            )
            embedding = embedding_response.data[0].embedding
            tokens = 0
            if embedding_response.usage:
                tokens += embedding_response.usage.total_tokens
            return Embedding(embeddings=embedding, tokens_used=tokens, dims=len(embedding))

    async def get_available_models(self):
        async with AsyncOpenAI(
            api_key=self.config.api_key,
            http_client=DefaultAioHttpClient(),
        ) as client:
            models = await client.models.list()
            return [ModelResponse(name=m.id) for m in models.data]

    async def _invoke_tools(
        self,
        tools: list[ToolDefinition],
        tool_calls: list[ToolCall],
    ) -> list[dict[str, str]]:
        
        tool_responses = []
        for tool_call in tool_calls:
            for tool in tools:
                if tool_call.name == tool.description["function"]["name"]:
                    tool_response = await tool.function(**tool_call.arguments)
                    tool_responses.append({
                        'tool': tool_call.name,
                        'response': tool_response}
                    )

        return tool_responses

    async def _get_tool_call_reason(
        self,
        client: AsyncOpenAI,
        messages: list[LLMMessage],
        tool_calls: list[ToolCall],
    ) -> Tuple[ChatCompletion, list[LLMMessage], LLMMessage]:
        prompt = f"""To execute this task, you will execute the following tool call: 
        {json.dumps([asdict(t) for t in tool_calls])}.
        Think about a concise reason (1-2 sentences max) why the tool call is necessary!
        """
        prompt = prepare_prompt(prompt)
        reason_prompt_message = LLMMessage(role='developer', content=prompt)
        new_messages = messages[:] + [reason_prompt_message]
        chat_completion = await client.chat.completions.create(
            messages=self._create_openai_messages(messages=new_messages),
            model=self.config.model
        )
        return chat_completion, new_messages, reason_prompt_message

    def _create_openai_messages(self, messages: list[LLMMessage]) -> list[ChatCompletionDeveloperMessageParam | ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam]:

        def transform(message: LLMMessage) -> ChatCompletionDeveloperMessageParam | ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam:
            if message.role == "assistant":
                return ChatCompletionAssistantMessageParam(role='assistant', content=message.content)
            elif message.role == "developer":
                return ChatCompletionDeveloperMessageParam(role='developer', content=message.content)
            elif message.role == "user":
                return ChatCompletionUserMessageParam(role='user', content=message.content)
            else:
                raise ValueError(f"Role: {message.role} not yet supported")

        return [transform(m) for m in messages]

    def _create_openai_tools(self, tool_description: ToolDescription) -> ChatCompletionToolParam:
        return ChatCompletionToolParam(
            type=tool_description["type"],
            function=tool_description["function"]
        )

    async def query(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] = [],
        force_tool_use: bool = False,
        response_format: Any | None = None,
    ) -> ChatResponse:
        async with AsyncOpenAI(
            api_key=self.config.api_key,
            http_client=DefaultAioHttpClient(),
        ) as client:
            chat_completion = await client.chat.completions.create(
                messages=self._create_openai_messages(messages=messages),
                model=self.config.model,
                tools=[self._create_openai_tools(tool.description) for tool in tools],
                response_format=type_to_response_format_param(response_format) if response_format else NotGiven # type: ignore[arg-type]
            )
            
            tokens = 0
            if chat_completion.usage:
                tokens = chat_completion.usage.total_tokens
            choice = chat_completion.choices[0]
            message = choice.message

            if choice.finish_reason == 'stop':
                # no tool calls requested by LLM but tool use forced.
                if not message.tool_calls and force_tool_use:
                    return ChatResponse(error='no_tool_found', tokens_used=tokens)

            # we are done
            if choice.finish_reason != 'tool_calls':
                return ChatResponse(content=message.content or "", tokens_used=tokens)

            # store all intermediate messages so that we can return them and 
            # eventuelly use them downstream if needed.
            tool_call_message_log: list[LLMMessage] = []

            # tool call branch
            tool_calls = [ToolCall(
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            ) for tc in message.tool_calls or []]

            for tool_call in tool_calls:
                tool_call_message_log.append(LLMMessage(role="assistant", content=f"I will call {asdict(tool_call)}"))

            reason_completion, reason_messages, reason_prompt_message = await self._get_tool_call_reason(client, messages, tool_calls)

            if reason_completion.usage:
                tokens += reason_completion.usage.total_tokens

            tool_call_message_log.append(reason_prompt_message)
            tool_call_message_log.append(LLMMessage(role="assistant", content=reason_completion.choices[0].message.content or ""))

            history = reason_messages[:] + [LLMMessage(
                role='assistant',
                content=reason_completion.choices[0].message.content or "",
            )]
        
            print("!!!!invoke tools")
            tool_responses = await self._invoke_tools(
                tools=tools,
                tool_calls=tool_calls,
            )
            print("!!!!tools invoked")

            prompt = f"""This is the result of the previous tool call(s): 
            {json.dumps(tool_responses)}.
            Provide a final answer to the user. Don't mention the tool call again.
            """
            prompt = prepare_prompt(prompt)

            final_answer_prompt_message = LLMMessage(role='developer', content=prompt)
            tool_call_message_log.append(final_answer_prompt_message)
            new_messages = history[:] + [final_answer_prompt_message]

            tool_completion = await client.chat.completions.parse(
                messages=self._create_openai_messages(messages=new_messages),
                model=self.config.model,
                response_format=response_format if response_format else NotGiven
            )

            if tool_completion.usage:
                tokens += tool_completion.usage.total_tokens

            return ChatResponse(
                content=tool_completion.choices[0].message.content or "",
                tool_call_log=tool_call_message_log,
                tokens_used=tokens,
            )