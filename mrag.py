from llm import LLMProvider, LLMMessage
from util import prepare_prompt
from pydantic import BaseModel

class KeyConceptResponse(BaseModel):
    keywords: list[str]
    context: str
    tags: list[str]

class MRag():

    _llm_provider: LLMProvider

    def __init__(self, llm_provider: LLMProvider):
        self._llm_provider = llm_provider

    async def generate_key_concepts(self, text_content: str) -> list[str]:
        assignment = """Generate a structured analysis of the following content by:
        1. Identifying the most salient keywords (focus on nouns, verbs, and
           key concepts)
        2. Extracting core themes and contextual elements
        3. Creating relevant categorical tags.
        Format the response as a JSON object:
        {
            "keywords": [ // several specific, distinct keywords that capture
                             key concepts and terminology 
                          // Order from most to least important
                          // Don’t include keywords that are the name of the speaker or time
                          // At least three keywords, but don’t be too redundant. ],
            "context": // one sentence summarizing:
                       // - Main topic/domain
                       // - Key arguments/points
                       // - Intended audience/purpose ,
            "tags": [ // several broad categories/themes for classification
                      // Include domain, format, and type tags
                      // At least three tags, but don’t be too redundant. ]
        }
        """
        assignment = prepare_prompt(assignment)

        prompt = f"""
        content: {text_content}
        """
        prompt = prepare_prompt(prompt)

        messages = [
            LLMMessage(role="developer", content=prompt),
            LLMMessage(role="user", content=prompt)
        ]

        keyconcepts = await self._llm_provider.query(
            messages=messages, 
            response_format=KeyConceptResponse,
            tools=[],
            force_tool_use=False,
            )
        print(keyconcepts)
