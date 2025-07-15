from llm import LLMProvider, LLMMessage, EmbeddingProvider, Embedding
from util import prepare_prompt
from pydantic import BaseModel
from chromadb.api import ClientAPI
from chromadb import Collection
from datetime import datetime
from json import JSONDecodeError
from uuid import uuid4, UUID
import numpy

type K = list[str]
type G = list[str]
type X = str

class KeyConceptResponse(BaseModel):
    keywords: K
    tags: G
    context: X

class MemoryNote(BaseModel):
    id: str
    ti: str         # timestamp
    ci: str         # content
    Ki: K           # keywords
    Gi: G           # tags
    Xi: X           # LLM-generated context
    ei: list[float] # embeddings
    Li: list[str]  # linked memories

class AMem():

    _llm_provider: LLMProvider
    _chroma_client: ClientAPI
    _embedding_provider: EmbeddingProvider
    _memory_notes: Collection | None

    def __init__(self, llm_provider: LLMProvider, chroma_client: ClientAPI, embedding_provider: EmbeddingProvider):
        self._llm_provider = llm_provider
        self._chroma_client = chroma_client
        self._embedding_provider = embedding_provider


    def setup_database(self):
        self._memory_notes = self._chroma_client.get_or_create_collection(
            name="memory_notes",
            embedding_function=None,
            metadata={
                "description": "A-Mem memory notes",
                "created": str(datetime.now())
            }
        )

    async def create_embedding(self, ci: str, Ki: K, Gi: G, Xi: X) -> Embedding:
        embedding_context = " ".join([ci] + Ki + Gi + [Xi])
        return await self._embedding_provider.create_embedding(embedding_context)

    def get_linked_memories(self, ei: list[float], k: int) -> list[MemoryNote]:
        assert(self._memory_notes)
        related = self._memory_notes.query(query_embeddings=ei, n_results=k)
        
        print("related", related)
        return []

    async def create_memory_note(self, ci: str, k: int = 10) -> MemoryNote:
        Ki, Gi, Xi = await self.generate_key_concepts(ci)
        embedding = await self.create_embedding(ci, Ki, Gi, Xi)
        ei = embedding.embeddings
        linked_memories = self.get_linked_memories(ei, k=k)
        Li = [mem.id for mem in linked_memories]
        
        note = MemoryNote(
            id=str(uuid4()),
            ti=str(datetime.now()),
            ci=ci,
            Ki=Ki,
            Gi=Gi,
            Xi=Xi,
            ei=ei,
            Li=Li,
        )

        return note

    def store_memory(self, note: MemoryNote):
        assert(self._memory_notes)
        # Each memory note mi in our collection M = {m1, m2, ..., mN }
        # is represented as:
        # mi = {ci, ti, Ki, Gi, Xi, ei, Li}
        # where
        #.   ci = original interaction content
        #    ti = timestamp of interaction
        #.   Ki = LLM-generated keywords
        #.   Gi = LLM-generated tags
        #.   Xi = LLm-generated contextual description
        #.   ei = embedding, where ei = enc[ concat(ci, Ki, Gi, Xi) ]
        #.   Li = set of linked memories that share semantic relationship
        self._memory_notes.add(
            ids=[note.id],
            embeddings=[numpy.array(note.ei)],
            documents=[note.model_dump_json()],
            metadatas=[{
                "Xi": note.Xi,
                "Gi": ",".join(note.Gi),
                "Ki": ",".join(note.Ki),
            }]
        )

    async def generate_key_concepts(self, ci: str) -> tuple[K, G, X]:
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
            "tags": [ // several broad categories/themes for classification
                      // Include domain, format, and type tags
                      // At least three tags, but don’t be too redundant. ],
            "context": // one sentence summarizing:
                       // - Main topic/domain
                       // - Key arguments/points
                       // - Intended audience/purpose
        }
        """
        assignment = prepare_prompt(assignment)

        prompt = f"""
        content: {ci}
        """
        prompt = prepare_prompt(prompt)

        messages = [
            LLMMessage(role="developer", content=prompt),
            LLMMessage(role="user", content=prompt)
        ]

        chat_response = await self._llm_provider.query(
            messages=messages, 
            response_format=KeyConceptResponse,
            tools=[],
            force_tool_use=False,
            )
        
        try:
            keyconcepts = KeyConceptResponse.model_validate_json(chat_response.content)
            return (keyconcepts.keywords, keyconcepts.tags, keyconcepts.context)
        except JSONDecodeError as e:
            raise RuntimeError(e)
