import asyncio
import os
from typing import Literal
import dotenv
import sys
from llm import OpenAIProvider, OpenAIConfig, ToolDefinition, LLMProvider
from agent import Agent, AgentDescription
from termcolor import cprint
from mcpcli import McpClient, get_server_parameters
from amem import AMem
from chromadb import EphemeralClient
import json

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


async def retrieve_email(tag: Literal['unread', 'none'] | None, sender: str | None) -> str: 
    print(f'CALL retrieve_email(tag={tag}, sender={sender})')
    if sender == 'mpfundstein@protonmail.com':
        return json.dumps([{
            "from": "mpfundstein@protonmail.com",
            "title": "WHERE IS MY CASH!",
            "content": "WHERE IS MY CASH YOU IDEOT!",
            "tags": ["important"]
        }])
    return json.dumps([
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
    ])

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

async def get_openai_provider() -> OpenAIProvider:
    chat_model = 'gpt-4o'
    emb_model = 'text-embedding-3-small'
    config = OpenAIConfig(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model=chat_model,
        embedding_model=emb_model,
    )
    llm_provider = OpenAIProvider(config)
    available_models = await llm_provider.get_available_models()

    chat_model_found = False
    emb_model_found = False
    for m in available_models:
        if m.name == chat_model:
            chat_model_found = True
        if m.name == emb_model:
            emb_model_found = True
        if chat_model_found and emb_model_found:
            break
    
    if not chat_model_found:
        raise RuntimeError(f"Chat model {chat_model} not available")
    if not emb_model_found:
        raise RuntimeError(f"Embedding model {emb_model} not available")

    return llm_provider

async def main_local_tools() -> int:

    llm_provider = await get_openai_provider()
    
    await test_tool_use(llm_provider=llm_provider)
    await test_memory(llm_provider=llm_provider)
    #await test_refusal(llm_provider=llm_provider)

    return 0

async def main_mcp_tools() -> int:

    llm_provider = await get_openai_provider()

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

async def test_mrag():
    llm_provider = await get_openai_provider()
    amem = AMem(llm_provider=llm_provider,
                chroma_client=EphemeralClient(),
                embedding_provider=llm_provider)
    amem.setup_database()

    cis = [
        "In the game Alpha Centauri, the spaceship is travelling to Alpha Omega",
        "De rampenbestrijding en de voorbereiding daarop is een taak van de gemeente en het bestuur van de veiligheidsregio.",
        "Om zich voor te kunnen bereiden op de rampenbestrijding heeft het bestuur van de veiligheidsregio informatie nodig van de exploitanten.",
        "De informatie die nodig is (opgesomd in Bijlage K) is in principe opgenomen in het VR. Deze bijlage handelt over het selecteren van rampscenario’s die de veiligheidsregio inzicht moeten geven in de dynamiek van effecten ten gevolge van een LOC. "
    ]

    for ci in cis:
        note = await amem.create_memory_note(ci)
        amem.store_memory(note=note)


    # Door deze informatie krijgt de veiligheidsregio een beeld van de mogelijke effecten buiten de Seveso-inrichting in geval van een ramp. Op basis van deze effecten bepaalt de veiligheidsregio, al dan niet in samenwerking met een bedrijfsbrandweer van de desbetreffende Seveso-inrichting, hoe moet worden opgetreden om de gevolgen van een ramp te minimaliseren." \
    #"Van belang is dat deze dynamiek (hiermee wordt bedoeld dat duidelijk is hoe het scenario zich ontwikkelt in de tijd) van de scenario’s is uitgewerkt. Een rampscenario kan instantaan optreden of zich langzaam ontwikkelen in de tijd. Indien het scenario zich langzaam ontwikkelt, kunnen nog maatregelen door de veiligheidsregio worden genomen, bijvoorbeeld het evacueren van personen in het effectgebied. De maatregelen van de veiligheidsregio kunnen dus worden afgestemd op het verloop van het rampscenario. Vandaar dat er informatie moet zijn over de ontwikkeling van een rampscenario. Daarom is in de in deze bijlage beschreven effectenboom aandacht gegeven aan het begrip ontwikkelingstijd."



    return 0


    text_content = """De veiligheidsregio bereidt zich voor op rampen op basis van artikel 17, eerste lid, van de Wet veiligheidsregio’s. Daarin staat vermeld:

Het bestuur van de veiligheidsregio stelt een rampbestrijdingsplan vast voor:

    locaties waarop een of meer bij algemene maatregel van bestuur aangewezen milieubelastende activiteiten worden verricht;
    inrichtingen en rampen die behoren tot een bij de maatregel, bedoeld onder a, aangewezen categorie;
    luchthavens die bij de maatregel, bedoeld onder a, zijn aangewezen.

In artikel 46, derde lid van de Wvr is het volgende bepaald:

Het bestuur van de veiligheidsregio draagt er zorg voor dat de bij de rampenbestrijding en de crisisbeheersing in de regio betrokken personen informatie wordt verschaft over de rampen en de crises die de regio kunnen treffen, de risico’s die hun inzet kan hebben voor hun gezondheid en de voorzorgsmaatregelen die in verband daarmee zijn of zullen worden getroffen.

in artikel 17, eerste lid, van de Wvr wordt verwezen naar een algemene maatregel van bestuur. Deze AMvB is het Besluit veiligheidsregio’s. In artikel 6.1.1, eerste lid, van het Bvr staat:

“Het bestuur van de veiligheidsregio stelt een rampbestrijdingsplan vast voor locaties waarop hogedrempelinrichtingen worden geëxploiteerd.”

De veiligheidsregio moet dus voor hogedrempeliginrichtingen overgaan tot het opstellen van een rampbestrijdingsplan. De informatie die het bestuur nodig heeft moet door degene die de hogedrempelinrichting exploiteert, worden aangeleverd op basis van artikel 48 van de Wet veiligheidsregio’s en de (deels) daarop gebaseerde paragraaf 4.2 van het Bal. Op grond van die informatie kan het bestuur van de veiligheidsregio alsnog besluiten dat voor de desbetreffende Seveso-inrichting geen rampbestrijdingsplan hoeft te worden vastgesteld.

Daarnaast bevat artikel 4.13, tweede lid, onder b, van het Bal bepalingen ten aanzien van het aanleveren van en informatie voor de externe hulpdiensten ten behoeve van het opstellen van rampbestrijdingsplannen."""

    await amem.generate_key_concepts(text_content)

if __name__ == "__main__":
    #rc = asyncio.run(main_local_tools())
    #rc = asyncio.run(main_mcp_tools())
    rc = asyncio.run(test_mrag())
    sys.exit(rc)
