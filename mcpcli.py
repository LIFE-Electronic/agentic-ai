from mcp import ClientSession, StdioServerParameters
from mcp.types import Tool, CallToolResult, TextContent
from mcp.client.stdio import stdio_client
from llm import ToolDefinition
from contextlib import _AsyncGeneratorContextManager
from typing import Any


def get_server_parameters() -> list[StdioServerParameters]:
    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-obsidian"],
        env={
            "OBSIDIAN_API_KEY": "c3f513fe9f29a37c1cacb43ad5629d59c435302c1c2cc55138220e4616571195"
        },
    )
    return [server_params]

class McpClient():

    _session: ClientSession | None = None
    _client: _AsyncGeneratorContextManager | None = None

    async def initialize(self, server_params: StdioServerParameters):
        assert(self._client is None)
        assert(self._session is None)
        self._client = stdio_client(server_params)
        read, write = await self._client.__aenter__()

        self._session = await ClientSession(read, write).__aenter__()
        await self._session.initialize()

    async def get_tools(self) -> list[ToolDefinition]:
        assert(self._session)
        tools = await self._session.list_tools()
        return [self._parse_mcp_tool(t) for t in tools.tools]

    async def cleanup(self):
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None

    def _parse_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        props: dict[str, Any] = {}
        for key, val in properties.items():
            #print(key, val)
            props[key] = {
                "type": val["type"],
                "description": val["description"]
            }

        return props

    def _parse_mcp_tool(self, mcp_tool: Tool) -> ToolDefinition:

        schema = mcp_tool.inputSchema
        properties = self._parse_properties(schema["properties"])
        required = schema["required"] if "required" in schema else []

        name = mcp_tool.name

        description = {
            "type": "function",
            "function": {
                "name": name,
                "description": mcp_tool.description,
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                }
            }
        }

        async def invoke_tool(**kwargs) -> str:
            if not self._session:
                return f"<tool_call_error>There was an error calling tool {name}: NO CONNECTION TO MCP SERVER</tool_call_error>"

            print(f"invoke MCP server tool: {name}")

            response: CallToolResult = await self._session.call_tool(name, **kwargs)
            if response.isError:
                return f"<tool_call_error>There was an error calling tool {name}: {response.content}</tool_call_error>"

            text_content = ""
            for content in response.content:
                if isinstance(content, TextContent):
                    text_content += content.text

            return text_content

        return ToolDefinition(
            description=description,
            function=invoke_tool,
        )
