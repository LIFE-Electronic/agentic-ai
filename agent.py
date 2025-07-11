from llm import LLMMessage, LLMProvider, Tool, ChatResponse
from pydantic.dataclasses import dataclass
from pydantic import BaseModel
from typing import Iterable, Literal
from memory import ShortTermMemoryProvider
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
    _stm_provider: ShortTermMemoryProvider
    _tools: list[Tool]
    _description: AgentDescription
    _force_tool_use: bool

    def __init__(
            self,
            description: AgentDescription,
            llm_provider: LLMProvider,
            stm_provider: ShortTermMemoryProvider | None,
            tools: Iterable[Tool] = [],
            force_tool_use: bool = False,
    ):
        self._llm_provider = llm_provider
        self._stm_provider = stm_provider
        self._description = description
        self._tools = tools[:]
        self._force_tool_use = force_tool_use

    def get_agent_description(self) -> AgentDescription:
        return self._description

    async def get_message_history(self) -> list[LLMMessage]:
        if self._stm_provider:
            return await self._stm_provider.retrieve_many()
        return []
    
    async def handle_user_message(
        self,
        message: str,
    ) -> AgentResponse:
        
        messages : list[LLMMessage] = []
        
        message_history : list[LLMMessage] = []
        if self._stm_provider:
            message_history = await self._stm_provider.retrieve_many()

        if not message_history:
            agent_desc = f"""
            You are an LLM-based agent. You execute tasks on behalf
            of your master. Always follow your agent description and
            the provided instructions. 
            Important: 
            
            <instructions>
            - If you have tools that you can use, use them. If you see
            in your message history that you have already used them, then
            don't use them again.
            - If you are assigned a task that does not fit
            your description, respond with an error as described in
            the output format.
            </instruction>
            <agent_description>
                <name>{self._description.name}</name>
                <description>{self._description}</description>
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