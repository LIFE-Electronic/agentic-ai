import asyncio
import os
from typing import Literal
import dotenv
import sys
from llm import OpenAIProvider, OpenAIConfig, LLMMessage, Tool, LLMProvider
from agent import Agent, AgentDescription
from memory import QueueBasedMemory

dotenv.load_dotenv()

async def retrieve_email(tag: Literal['unread'] | None): 
    print('retrieve_email: ', tag)
    return [
        {
            "from": "loverboy@gmail.com",
            "title": "Meeting tonight?",
            "content": "Shall I come over tonight? Don't tell your man!!!!"
        },
        {
            "from": "wife@gmail.com",
            "title": "CALL ME!",
            "content": "Hey! We need to talk. It's serious! CALL ME!"
        }
    ]

async def test_tool_use(llm_provider: LLMProvider):
    
    tools = [Tool(description={
        "type": "function",
        "function": {
            "name": "retrieve_emails",
            "description": "Retrievs all email for today.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "The tag to filter the query. Can be 'unread'."
                    }
                },
                "required": ["tag"],
                "additionalProperties": False
            }
        }
    }, function=retrieve_email)]


    desc = AgentDescription(
        name="Max",
        description="You are a helpful assistant that can read emails. " \
        "When multiple emails are available, summarize them into a small story."
        "Answer the users query faithfully. Be concise and direct in your answers. Don't talk too much!"
    )
    agent = Agent(
        description=desc,
        llm_provider=llm_provider,
        stm_provider=QueueBasedMemory(),
        tools=tools,
    )

    prompt = "What new emails do I have today?"
    print(f"-----> USER: {prompt}")

    response = await agent.handle_user_message(prompt)

    print(f"-----> AGENT: {response.content}")

    prompt = "Does my wife know?"
    print(f"-----> USER: {prompt}")

    response = await agent.handle_user_message(prompt)

    print(f"-----> AGENT: {response.content}")

    print('\n\n------')
    for x in await agent.get_message_history():
        print(x)
    print('\n\n------')

    desc = AgentDescription(
        name="Moritz",
        description="You are a helpful assistant that can read emails. Answer the users query faithfully."
    )
    agent = Agent(
        description=desc,
        llm_provider=llm_provider,
        stm_provider=None,
        tools=tools,
        force_tool_use=True,
    )

    response = await agent.handle_user_message('Write an entry into my diary with content "School sucls"')
    print(response.error)

async def test_memory(llm_provider: LLMProvider):

    stm_memory = QueueBasedMemory()

    desc = AgentDescription(
        name="Boris",
        description="You are a personal diary agent. You keep track of what is done and you answer queries when asked. " \
        "When the user tells you something, you summarize it in one sentence, and you respond and acknowledge what it was." \
        "Return your response in plain text. Don't wrap it in <memory>")

    agent = Agent(
        description=desc,
        llm_provider=llm_provider,
        stm_provider=stm_memory,
    )

    agent_response = await agent.handle_user_message("Today I was in school!")

    agent_response = await agent.handle_user_message("Then I went to the bank")

    agent_response = await agent.handle_user_message("Finally I slept")

    agent_response = await agent.handle_user_message("What did I do today?")
    
    for x in await agent.get_message_history():
        print(x)
    
    print(f"- {agent_response.content}")

    agent_response = await agent.handle_user_message("Can you read my emails?")
    print(f"- {agent_response}")

async def main() -> int:
    model = 'gpt-4o'
    config = OpenAIConfig(api_key=os.environ.get("OPENAI_API_KEY"), model=model)
    llm_provider = OpenAIProvider(config)
    available_models = await llm_provider.get_available_models()

    found = False
    for m in available_models:
        if m.name == model:
            found = True
            break
    if not found:
        print(f"Model {model} not available")
        return 1
    
    await test_tool_use(llm_provider=llm_provider)
    #await test_memory(llm_provider=llm_provider)

    return 0

if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
