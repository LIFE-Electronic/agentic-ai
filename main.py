import asyncio
import os
from typing import Literal
import dotenv
import sys
from llm import OpenAIProvider, OpenAIConfig, ToolDefinition, LLMProvider
from agent import Agent, AgentDescription
from termcolor import cprint
from mcpcli import McpClient, get_server_parameters

dotenv.load_dotenv()

async def eval_agent(agent: Agent, user_messages: list[str], print_history: bool = False):
    for user_message in user_messages:
        cprint(f"-----> USER: {user_message}", "yellow")
        response = await agent.handle_user_message(user_message)
        if response.error:
            cprint(f"Error: {response.error}", "red")
            if response.error == 'agent_refusal_of_task':
                cprint(f"Agent refused task: {response.refusal_reason}", "red")
        cprint(f"-----> AGENT: {response.content}", "green")

    if print_history:
        print("----------")
        for x in await agent.get_message_history():
            color = "green"
            if x.role == "developer":
                color = "cyan"
            elif x.role == "user":
                color = "yellow"
            cprint(f"- {x}", color)
        print("----------")

    print(f"tokens used: {agent.get_tokens_used()}")


async def retrieve_email(tag: Literal['unread', 'none'] | None, sender: str | None): 
    print(f'CALL retrieve_email(tag={tag}, sender={sender})')
    if sender == 'mpfundstein@protonmail.com':
        return [{
            "from": "mpfundstein@protonmail.com",
            "title": "WHERE IS MY CASH!",
            "content": "WHERE IS MY CASH YOU IDEOT!",
            "tags": ["important"]
        }]
    return [
        {
            "from": "loverboy@gmail.com",
            "title": "Meeting tonight?",
            "content": "Shall I come over tonight? Don't tell your man!!!!",
            "tags": ["unread"]
        },
        {
            "from": "wife@gmail.com",
            "title": "CALL ME!",
            "content": "Hey! We need to talk. It's serious! CALL ME!",
            "tags": ["unread"]
        }
    ]

async def test_tool_use(llm_provider: LLMProvider):
    
    tools: list[ToolDefinition] = [ToolDefinition(description={
        "type": "function",
        "function": {
            "name": "retrieve_emails",
            "description": "Retrieves emails from email server. Use tag or from_filter to search.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "The tag to filter the query. Can be 'unread' or 'none'."
                    },
                    "sender": {
                        "type": "string",
                        "description": "Retrieve only emails that are from the specificed sender. Use it if you want to search for emails from a specific sender. Otherwise leave empty"
                    }
                },
                "required": ["tag", "sender"],
                "additionalProperties": False
            }
        }
    }, function=retrieve_email)]


    desc = AgentDescription(
        name="Max",
        description="You are a helpful assistant that manages my emails with the email tool." \
        "When multiple emails are available, summarize them into a small story."
        "Answer the users query faithfully. Be concise and direct in your answers. Don't talk too much!"
    )
    agent = Agent(
        description=desc,
        llm_provider=llm_provider,
        tools=tools,
    )
    prompts: list[str] = [
        #"What tools can you use??",
        "What new emails do I have today?",
        "Does my wife know?",
        "Get me all emails from sender: mpfundstein@protonmail.com",
    ]
    
    await eval_agent(agent, prompts, print_history=True)

async def test_refusal(llm_provider: LLMProvider):

    tools = [ToolDefinition(description={
        "type": "function",
        "function": {
            "name": "retrieve_emails",
            "description": "Retrieves emails from email server. Use tag or from_filter to search.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "The tag to filter the query. Can be 'unread' or 'none'."
                    },
                    "sender": {
                        "type": "string",
                        "description": "Retrieve only emails that are from the specificed sender. Use it if you want to search for emails from a specific sender. Otherwise leave empty"
                    }
                },
                "required": ["tag", "sender"],
                "additionalProperties": False
            }
        }
    }, function=retrieve_email)]

    desc = AgentDescription(
        name="Moritz",
        description="You are a helpful assistant that can read emails. Answer the users query faithfully."
    )
    agent = Agent(
        description=desc,
        llm_provider=llm_provider,
        tools=tools,
        force_tool_use=False,
    )

    messages = [
        'Write an entry into my diary with content "School sucls"'
    ]

    await eval_agent(agent, messages, print_history=True)

async def test_memory(llm_provider: LLMProvider):

    desc = AgentDescription(
        name="Boris",
        description="You are a personal diary agent. You keep track of what is done and you answer queries when asked. " \
        "When the user tells you something, you summarize it in one sentence, and you respond and acknowledge what it was." \
        "Return your response in plain text.")

    agent = Agent(
        description=desc,
        llm_provider=llm_provider,
    )

    prompts: list[str] = [
        "Today I was in school!",
        "Then I went to the bank",
        "Finally I slept",
        "What did I do today?",
        "Can you read my emails?"
    ]
    
    await eval_agent(agent, prompts, print_history=True)

async def get_llm_provider() -> LLMProvider:
    model = 'gpt-4o'
    config = OpenAIConfig(api_key=os.environ.get("OPENAI_API_KEY", ""), model=model)
    llm_provider = OpenAIProvider(config)
    available_models = await llm_provider.get_available_models()

    found = False
    for m in available_models:
        if m.name == model:
            found = True
            break
    if not found:
        raise RuntimeError(f"Model {model} not available")

    return llm_provider

async def main_local_tools() -> int:

    llm_provider = await get_llm_provider()
    
    await test_tool_use(llm_provider=llm_provider)
    await test_memory(llm_provider=llm_provider)
    #await test_refusal(llm_provider=llm_provider)

    return 0

async def main_mcp_tools() -> int:

    llm_provider = await get_llm_provider()

    params = get_server_parameters()

    mcpclient = McpClient()
    print("start MCP")
    try:
        await mcpclient.initialize(params[0])
        print("Done")
        tools = await mcpclient.get_tools()
        tools = tools[:-9] + tools[9:]
    
        agent_desc = AgentDescription(
            name="Max",
            description="You are a helpful assistant. When the user requests something, check which tools" \
            "you can use and execute them."
        )
        agent = Agent(
            description=agent_desc,
            llm_provider=llm_provider,
            tools=tools,
        )

        messages = [
            "Which tools do you know?",
            "Search files regarding WolperTec"
        ]

        await eval_agent(agent=agent, user_messages=messages, print_history=True)

    finally:
        await mcpclient.cleanup()

    return 0

if __name__ == "__main__":
    #rc = asyncio.run(main_local_tools())
    rc = asyncio.run(main_mcp_tools())
    sys.exit(rc)
