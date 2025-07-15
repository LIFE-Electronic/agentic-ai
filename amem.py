from llm import LLMProvider, LLMMessage, EmbeddingProvider, Embedding
from util import prepare_prompt
from pydantic import BaseModel
from typing import Literal
from chromadb.api import ClientAPI
from chromadb import Collection
from datetime import datetime
from json import JSONDecodeError
from uuid import uuid4, UUID
import numpy
import graphviz # type: ignore[import-untyped]

type K = list[str]
type G = list[str]
type X = str

class KeyConceptResponse(BaseModel):
    keywords: K
    tags: G
    context: X

class LinkMemoryResponse(BaseModel):
    links: list[str] # ids of linked memories

class EvolveMemoryNoteResponse(BaseModel):
    should_evolve: bool
    actions: Literal['strengthen', 'merge', 'prune']
    suggested_connections: list[str]
    tags_to_update: list[str]

class MemoryNote(BaseModel):
    id: str
    ti: str         # timestamp
    ci: str         # content
    Ki: K           # keywords
    Gi: G           # tags
    Xi: X           # LLM-generated context
    ei: list[float] # embeddings
    Li: list[str]  # linked memories

    def get_prompt_format(self) -> str:

        lis = "".join([f"<id>{li}</id>" for li in self.Li])

        return prepare_prompt(f"""
        <memory>
            <id>{self.id}</id>
            <timestamp>{self.ti}</timestamp>
            <context>{self.Xi}</context>
            <content>{self.ci}</content>
            <keywords>{self.Ki}</keywords>
            <tags>{self.Gi}</tags>
            <linked_memories>
                {lis}
            </linked_memories>
        </memory>
        """)

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
                "created": str(datetime.now()),
                "hnsw:space": "cosine"
            }
        )

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
        self._memory_notes.upsert(
            ids=[note.id],
            embeddings=[numpy.array(note.ei)],
            documents=[note.model_dump_json()],
            metadatas=[{
                "ti": note.ti,
                "Xi": note.Xi,
                "Gi": ",".join(note.Gi),
                "Ki": ",".join(note.Ki),
                "Li": ",".join(note.Li)
            }]
        )

    def draw_memory_graph(self, output_file: str, max_notes: int = 10000) -> None:
        assert(self._memory_notes)
        notes = self._memory_notes.get(limit=max_notes)

        docs = notes.get("documents", [])
        if not docs:
            return
        
        dot = graphviz.Digraph(comment=f'A-mem: {output_file}')
        
        already_drawn: set[str] = set()
        for doc in docs:
            note = MemoryNote.model_validate_json(doc)
            if note.id not in already_drawn:
                dot.node(name=note.id, label=note.ci[0:10])
                already_drawn.add(note.id)
            for linked_notes in note.Li:
                dot.edge(note.id, linked_notes)

        dot.render(output_file).replace('\\', '/')


    async def _create_embedding(self, ci: str, Ki: K, Gi: G, Xi: X) -> Embedding:
        embedding_context = " ".join([ci] + Ki + Gi + [Xi])
        return await self._embedding_provider.create_embedding(embedding_context)

    def _get_nearest_memory_notes(self, ei: list[float], k: int, cos_threshold: float) -> list[MemoryNote]:
        assert(self._memory_notes)
        related = self._memory_notes.query(query_embeddings=ei, n_results=k)

        first_docs = related.get('documents', [])
        first_dist = related.get('distances', [])
        if not first_docs or not first_dist:
            return []

        docs = first_docs[0]
        distances = first_dist[0]

        filtered_docs: list[MemoryNote] = []
        for i, doc in enumerate(docs):
            dist = distances[i]
            if dist > cos_threshold:
                continue
            
            memory_note = MemoryNote.model_validate_json(doc)
            filtered_docs.append(memory_note)

        return filtered_docs

    async def create_memory_note(self, ci: str, k: int = 10) -> tuple[MemoryNote, list[MemoryNote]]:

        tokens = 0

        Ki, Gi, Xi, tokens_used = await self._generate_key_concepts(ci)

        tokens += tokens_used

        embedding = await self._create_embedding(ci, Ki, Gi, Xi)
        ei = embedding.embeddings
        
        tokens += embedding.tokens_used

        note = MemoryNote(
            id=str(uuid4()),
            ti=str(datetime.now()),
            ci=ci,
            Ki=Ki,
            Gi=Gi,
            Xi=Xi,
            ei=ei,
            Li=[],
        )

        Mnear = self._get_nearest_memory_notes(ei, k=k, cos_threshold=1.0)
        updates_notes: list[MemoryNote] = []
        if Mnear:
            link_memory_response, tokens_used = await self._get_links_for_new_memory(note, Mnear)

            tokens += tokens_used

            note.Li = link_memory_response.links

            # go through all neighbours that are linked and make sure they obtain a link
            # to the new memory as well
            for link in link_memory_response.links:
                for mn in Mnear:
                    if mn.id == link:
                        print(f"update memory: {mn.id}")
                        if note.id not in mn.Li:
                            print(f"    add link to: {note.id}")
                            mn.Li.append(note.id)
                            updates_notes.append(mn)
                        break

            # write to db

        print(f"created memory note: {note.id}. tokens_used: {tokens}. Links: {note.Li}")

        return note, updates_notes
    
    #async def _evolve_memory_note(self, note: MemoryNote, Mnear: list[MemoryNote]) -> tuple[EvolveMemoryNoteResponse, int]:
    #    return [,0]

    async def _get_links_for_new_memory(self, note: MemoryNote, Mnear: list[MemoryNote]) -> tuple[LinkMemoryResponse, int]:

        new_mem_str = note.get_prompt_format()
        Mnear_str = "\n".join([m_n.get_prompt_format() for m_n in Mnear])

        task = """
        You are an AI memory evolution agent responsible for managing and
        evolving a knowledge base.
        You will be given a new memory note that contains information such as
        its context, content, keywords and tags. You will also given a list of
        nearest neighbours memory notes that have a high similarity with the new memory note.
        Based on this information, determine to which of the nearest neighbours memory notes
        (if any at all) the new memory should be directly linked to."
        Format the response as a JSON object:
        {
            "links": [ // ids of memories that the new memory should link to ]
        }"""
        task = prepare_prompt(task)

        prompt = f"""
        <new_memory>
        {new_mem_str}
        </new_memory>
        <nearest_memories>
        {Mnear_str}
        </nearest_memories>
        """
        task = prepare_prompt(task)
        
        messages = [
            LLMMessage(role='developer', content=task),
            LLMMessage(role='user', content=prompt)
        ]

        chat_response = await self._llm_provider.query(
            messages=messages,
            tools=[],
            force_tool_use=False,
            response_format=LinkMemoryResponse,
        )

        return LinkMemoryResponse.model_validate_json(chat_response.content), chat_response.tokens_used

    async def _generate_key_concepts(self, ci: str) -> tuple[K, G, X, int]:
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
            return (keyconcepts.keywords, keyconcepts.tags, keyconcepts.context, chat_response.tokens_used)
        except JSONDecodeError as e:
            raise RuntimeError(e)
