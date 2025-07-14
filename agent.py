from llm import LLMMessage, LLMProvider, ToolDefinition, ChatResponse
from pydantic.dataclasses import dataclass
from pydantic import BaseModel
from typing import Iterable, Literal
from memory import MemoryProvider, RAMMemory
from util import prepare_prompt

@dataclass
class AgentDescription:
    name: str
    description: str

class AgentResponseFormat(BaseModel):
    response: str
    refusal_reason: Literal['cant_handle_task'] | None = None

@dataclass
class AgentResponse:
    content: str
    refusal_reason: Literal['cant_handle_task'] | None = None
    error: Literal['no_tool_found', 'agent_refusal_of_task'] | None = None

class Agent():

    _llm_provider: LLMProvider
    _stm_provider: MemoryProvider
    _tools: list[ToolDefinition]
    _description: AgentDescription
    _force_tool_use: bool
    _tokens_used: int

    def __init__(
            self,
            description: AgentDescription,
            llm_provider: LLMProvider,
            stm_provider: MemoryProvider | None = None,
            tools: list[ToolDefinition] = [],
            force_tool_use: bool = False,
    ):
        self._llm_provider = llm_provider
        self._stm_provider = stm_provider
        self._description = description
        self._tools = tools
        self._force_tool_use = force_tool_use
        self._tokens_used = 0
        self._stm_provider = RAMMemory()

    def get_agent_description(self) -> AgentDescription:
        return self._description

    def get_tokens_used(self) -> int:
        return self._tokens_used

    async def get_message_history(self, limit : int | None = None) -> list[LLMMessage]:
        if self._stm_provider:
            return await self._stm_provider.retrieve_many(limit=limit)
        return []
    
    async def handle_user_message(
        self,
        message: str,
    ) -> AgentResponse:
        
        messages : list[LLMMessage] = []
        
        message_history : list[LLMMessage] = []
        if self._stm_provider:
            message_history = await self._stm_provider.retrieve_many(limit=None)

        if not message_history:
            agent_desc = f"""
            You are an LLM-based agent. You execute tasks on behalf
            of your master. Always follow your agent description and
            the provided instructions. 
            Important: 
            
            <instructions>
            - If you have tools that you can use, use them.
            - If you are assigned a task that does not fit
            your description, respond with an error as described in
            the output format.
            - If you refuse a task, always provide a concise explanation why.
            Don't add anything else, just the reason of why you refused it.
            </instruction>
            <agent_description>
                <name>{self._description.name}</name>
                <description>{self._description.description}</description>
            </agent_description>
            """
            agent_desc = prepare_prompt(agent_desc)
            developer_message = LLMMessage(role='developer', content=agent_desc)
            messages += [developer_message]
            if self._stm_provider:
                await self._stm_provider.store(developer_message)
            
        user_message = LLMMessage(role='user', content=message)

        messages += message_history[:] + [user_message]

        chat_response = await self._llm_provider.query(
            messages=messages,
            tools=self._tools,
            force_tool_use=self._force_tool_use,
            response_format=AgentResponseFormat
        )

        self._tokens_used += chat_response.tokens_used

        if chat_response.error:
            return AgentResponse(content='', error=chat_response.error)

        agent_response = AgentResponseFormat.model_validate_json(chat_response.content)
        content = agent_response.response

        if self._stm_provider:
            assistent_message = LLMMessage(role='assistant', content=content)

            await self._stm_provider.store(user_message)

            for tool_call_log_message in chat_response.tool_call_log:
                await self._stm_provider.store(tool_call_log_message)

            await self._stm_provider.store(assistent_message)

        return AgentResponse(
            content=content,
            refusal_reason=agent_response.refusal_reason,
            error='agent_refusal_of_task' if agent_response.refusal_reason else None
        )