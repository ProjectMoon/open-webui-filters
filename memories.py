"""
title: Memory Filter
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 0.0.1
required_open_webui_version: 0.3.8
"""

# System imports
import asyncio
import hashlib
import uuid
import json

from typing import Optional, List, Dict, Callable, Any, NewType, Tuple, Awaitable
from typing_extensions import TypedDict, NotRequired

# Libraries available to OpenWebUI
import markdown
from bs4 import BeautifulSoup
from pydantic import BaseModel as PydanticBaseModel, Field
import chromadb
from chromadb import Collection as ChromaCollection
from chromadb.api.types import Document as ChromaDocument

# OpenWebUI imports
from config import CHROMA_CLIENT
from apps.rag.main import app
from utils.misc import get_last_user_message, get_last_assistant_message
from main import generate_chat_completions

# OpenWebUI aliases
EMBEDDING_FUNCTION = app.state.EMBEDDING_FUNCTION

# Custom type declarations
EmbeddingFunc = NewType('EmbeddingFunc', Callable[[str], List[Any]])

# Prompts
ENRICHMENT_SUMMARY_PROMPT = """
You are tasked with analyzing the following Characters and Plot Details
sections and reducing this set of information into lists of the most
important points needed for the continuation of the narrative you are
writing. Remove duplicate or conflicting information. If there is conflicting
information, decide on something consistent and interesting for the story.

Your reply must consist of two sections: Characters and Plot Details. These
sections must be markdown ### Headers. Under each header, respond with a
list of bullet points. Each bullet point must be one piece of relevant information.

Limit each bullet point to one sentence. Respond ONLY with the Characters and
Plot Details sections, with the bullet points under them, and nothing else.
Do not respond with any commentary. ONLY respond with the bullet points.
""".replace("\n", " ").strip()

QUERY_PROMPT = """
You are tasked with generating questions for a vector database
about the narrative presented below. The queries must be questions about
parts of the story that you need more details on. The questions must be
about past events in the story, or questions about the characters involved
or mentioned in the scene (their appearance, mental state, past actions, etc).

Your reply must consist of two sections: Characters and Plot Details. These
sections must be markdown ### Headers. Under each header, respond with a
list of bullet points. Each bullet point must be a single question or sentence
that will be given to the vector database. Generate a maximum of 5 Character
queries and 5 Plot Detail queries.

Limit each bullet point to one sentence. Respond ONLY with the Characters and
Plot Details sections, with the bullet points under them, and nothing else.
Do not respond with any commentary. ONLY respond with the bullet points.
""".replace("\n", " ").strip()

SUMMARIZER_PROMPT = """
You are a narrative summarizer. Summarize the given message as if it's
part of a story. Your response must have two separate sections: Characters
and Plot Details. These sections should be markdown ### Headers. Under each
section, respond with a list of bullet points. This knowledge will be stored
in a vector database for your future use.

The Characters section should note any characters in the scene, and important
things that happen to them. Describe the characters' appearances, actions,
mental states, and emotional states. The Plot Details section should have a
list of important plot details in this scene.

The bullet points you generate must be in the context of storing future
knowledge about the story. Do not focus on useless details: only focus on
information that you could lose in the future as your context window shifts.

Limit each bullet point to one sentence. The sentence MUST be in the PAST TENSE.
Respond ONLY with the Characters and Plot Details sections, with the bullet points
under them, and nothing else. Do not respond with any commentary. ONLY respond with
the bullet points.
""".replace("\n", " ").strip()

class Message(TypedDict):
    id: NotRequired[str]
    role: str
    content: str

class MessageInsertMetadata(TypedDict):
    role: str
    chapter: str

class MessageInsert(TypedDict):
    message_id: str
    content: str
    metadata: MessageInsertMetadata
    embeddings: List[Any]


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

class SummarizerResponse(BaseModel):
    characters: List[str]
    plot: List[str]


class Summarizer(BaseModel):
    message: str
    model: str
    prompt: str = SUMMARIZER_PROMPT

    def extract_section(self, soup: BeautifulSoup, section_name: str) -> List[str]:
        for h3 in soup.find_all('h3'):
            heading = h3.get_text().strip()
            if heading != section_name:
                continue

            # Find the next sibling which should be a <ul> or <ol>
            ul = h3.find_next_sibling('ul')
            ol = h3.find_next_sibling('ol')
            list_items = []

            if ul:
                list_items = [li.get_text().strip() for li in ul.find_all('li')]
            elif ol:
                list_items = [li.get_text().strip() for li in ol.find_all('li')]

            return list_items
        return []

    def sanitize_section(self, bullet_points: List[str]) -> List[str]:
        return [
            bullet.strip().lstrip("-*•123456789").strip() for bullet in bullet_points
        ]

    async def summarize(self) -> SummarizerResponse:
        messages: List[Message] = [
            { "role": "system", "content": SUMMARIZER_PROMPT },
            { "role": "user", "content": self.message }
        ]

        request = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10s"
        }

        resp = await generate_chat_completions(request)
        if "choices" in resp and len(resp["choices"]) > 0:
            content: str = resp["choices"][0]["message"]["content"]
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, "html.parser")
            character_results = self.extract_section(soup, "Characters")
            character_results = self.sanitize_section(character_results)
            plot_points = self.extract_section(soup, "Plot Details")
            plot_points = self.sanitize_section(plot_points)

            return SummarizerResponse(characters=character_results, plot=plot_points)
        else:
            return SummarizerResponse(characters=[], plot=[])

class Chapter(BaseModel):
    """
    Focuses on a single 'chapter,' or chunk of a conversation. Provides methods to
    search for data in this section of conversational story history.
    """

    convo_id: Optional[str]
    client: chromadb.ClientAPI
    chapter_id: str
    messages: List[Message]
    embedding_func: EmbeddingFunc

    def create_metadata(self) -> Dict:
        return { "convo_id": self.convo_id, "chapter": self.chapter_id }

    def get_collection(self) -> Optional[ChromaCollection]:
        try:
            coll = self.client.get_collection("stories")

            if not self.convo_id:
                self.convo_id = (
                    coll.metadata["current_convo_id"] if "current_convo_id" in coll.metadata else None
                )

            return coll
        except ValueError as e:
            return None


    def _create_inserts(self, summary: SummarizerResponse) -> List[MessageInsert]:
        inserts = []
        plot_points = summary.plot
        character_points = summary.characters

        for plot_point in plot_points:
            inserts.append({
                'id': str(uuid.uuid4()),
                'content': plot_point,
                'metadata': {
                    "convo_id": self.convo_id,
                    "chapter": self.chapter_id,
                    "type": "plot"
                },
                'embedding': self.embedding_func(plot_point)
            })

        for character_point in character_points:
            inserts.append({
                'id': str(uuid.uuid4()),
                'content': character_point,
                'metadata': {
                    "convo_id": self.convo_id,
                    "chapter": self.chapter_id,
                    "type": "character"
                },
                'embedding': self.embedding_func(character_point)
            })

        return inserts


    def chapter_state(self) -> dict:
        """Useful for storing current place in chapter, and convo switching."""
        coll = self.get_collection()
        result = coll.get(ids=f"chapter-{self.chapter_id}", include=["metadatas"])
        if len(result.metadatas) > 0:
            return result.metadatas[0]
        else:
            return {}


    def embed(self, summary: SummarizerResponse):
        """
        Store plot points for this chapter in ChromaDB.
        """
        coll = self.get_collection()
        if not self.convo_id:
            return

        inserts = self._create_inserts(summary)

        if len(inserts) > 0:
            documents = [entry['content'] for entry in inserts]
            metadatas = [entry['metadata'] for entry in inserts]
            ids = [entry['id'] for entry in inserts]
            embeddings = [entry['embedding'] for entry in inserts]
            coll.upsert(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)

    def query_plot(self, search_term):
        return self.query(search_term, "plot")

    def query_characters(self, search_term):
        return self.query(search_term, "character")

    def query(self, search_term: str, type: str) -> List[ChromaDocument]:
        coll = self.get_collection()
        if coll and self.convo_id:
            term_embedding = self.embedding_func(search_term)
            results = coll.query(
                query_embeddings=[term_embedding],
                include=["documents", "metadatas"],
                where={
                    "$and": [
                        { "convo_id": self.convo_id },
                        { "chapter": self.chapter_id },
                        { "type": type }
                    ]
                },
                n_results = 5
            )

            # flatten out list of list of documents
            # because chroma returns a List[List[Document]] for some reason.
            if 'documents' in results:
                docs = [
                    doc
                    for doc_list in results['documents']
                    for doc in doc_list
                ]

                metadatas = [
                    md
                    for md_list in results['metadatas']
                    for md in md_list
                ]

                results = []
                for (doc, metadata) in zip(docs, metadatas):
                    results.append({ "doc": doc, "metadata": metadata })

                return results
            else:
                return []
        else:
            return []


class Story(BaseModel):
    """Container for chapters. Manages an entire conversation."""

    convo_id: Optional[str] = None
    client: chromadb.ClientAPI
    messages: List[Message]
    embedding_func: EmbeddingFunc

    def _collection_name(self):
        return f"stories"

    def create_metadata(self):
        try:
            coll = self.client.get_collection(self._collection_name())
            if coll:
                # If we have pre-specified a convo id, update metadata
                # of collection accordingly.
                if self.convo_id:
                    metadata = coll.metadata
                    metadata['current_convo_id'] = self.convo_id
                    metadata["hnsw:space"] = "cosine"
                    coll = self.client.get_or_create_collection(
                        name=self._collection_name(), metadata=metadata
                    )
                else: # Otherwise pull it out of the database.
                    self.convo_id = (
                        coll.metadata['current_convo_id'] if 'current_convo_id' in coll.metadata else None
                    )

            return coll.metadata
        except ValueError:
            return { "current_convo_id": "<unset>", "current_chapter": 1 }

    def convo_state(self) -> dict:
        """Retrieve information about the current conversation."""
        if not self.convo_id or self.convo_id == "<unset>":
            return {}

        convo_state_id = f"convo-{self.convo_id}"
        coll = self.get_collection()
        result = coll.get(ids=[convo_state_id], include=["metadatas"])
        if len(result.metadatas) > 0:
            return result.metadatas[0]
        else:
            # insert convo state
            # TODO do something useful with convo summary
            convo_summary = f"State for convo {self.convo_id}"
            convo_metadata = { "current_chapter": 1 }

            coll.add(
                ids=[convo_state_id],
                documents=[convo_summary], # maybe store convo summary here?
                embeddings=self.embedding_func(convo_summary),
                metadatas=[convo_metadata]
            )

            return convo_metadata

    def switch_convo(self):
        """Force a switch of current conversation."""
        if not self.convo_id:
            # If we have only a user message (i.e. start of
            # conversation), forcibly set to <unset>
            if len(self.messages) < 2:
                self.convo_id = "<unset>"
            else:
                # Otherwise attempt to get the cllection, which forces
                # metatada creation and updates.
                self.get_collection()

    def get_collection(self):
        """Retrieve the collection, with its context set to the current convo ID."""
        try:
            coll = self.client.get_collection(self._collection_name())
            if coll:
                # If we have pre-specified a convo id, update metadata
                # of collection accordingly.
                if self.convo_id:
                    metadata = coll.metadata
                    metadata['current_convo_id'] = self.convo_id
                    metadata["hnsw:space"] = "cosine"
                    return self.client.get_or_create_collection(
                        name=self._collection_name(), metadata=metadata
                    )
                else: # Otherwise pull existing convo id out of the database.
                    self.convo_id = (
                        coll.metadata['current_convo_id'] if 'current_convo_id' in coll.metadata else None
                    )

                return coll
        except ValueError:
            # if the stories collection does not exist, create it
            # completely from scratch.
            metadata = { "current_convo_id": "<unset>", "hnsw:space": "cosine" }
            return self.client.get_or_create_collection(self._collection_name(), metadata=metadata)

    def _current_chapter(self) -> int:
        try:
            return self.convo_state()["current_chapter"]
        except:
            return 1

    def _current_chapter_object(self) -> Chapter:
        return Chapter(
            convo_id = self.convo_id, chapter_id=str(self._current_chapter()),
            messages=self.messages, client=self.client, embedding_func=self.embedding_func
        )

    def embed_summary(self, summary: SummarizerResponse):
        self._current_chapter_object().embed(summary)

    def query_plot(self, term: str) -> List[ChromaDocument]:
        return self._current_chapter_object().query_plot(term)

    def query_characters(self, term: str) -> List[ChromaDocument]:
        return self._current_chapter_object().query_characters(term)


# Utils
def create_enrichment_summary_prompt(
        narrative: str,
        character_details: List[str],
        plot_details: List[str]
) -> str:
    prompt = ENRICHMENT_SUMMARY_PROMPT
    prompt += "Here are the original Character and Plot Details sections."
    prompt += " Summarize them according to the instructions.\n\n"

    snippets = "##  Character Details:\n"
    for character_detail in character_details:
        snippets += f"- {character_detail}\n"

        snippets = snippets.strip()
        snippets += "\n"

    snippets += "\n\n## Plot Details:\n"
    for plot_point in plot_details:
        snippets += f"- {plot_point}\n"

        snippets = snippets.strip()
        snippets += "\n"

    snippets = snippets.strip()
    prompt += snippets + "\n\n"


    prompt += "Additionally, the narrative you must continue is provided below."
    prompt += "\n\n-----\n\n"
    prompt += narrative
    return prompt.strip()


def create_context(results: SummarizerResponse) -> Optional[str]:
    if not results:
        return None

    character_details = results.characters
    plot_details = results.plot

    snippets = "## Relevant Character Details:\n"
    snippets += "These are relevant bits of information about characters in the story.\n"

    for character_detail in character_details:
        snippets += f"- {character_detail}\n"

        snippets = snippets.strip()
        snippets += "\n"

    snippets += "\n\n## Relevant Plot Details:\n"
    snippets += "These are relevant plot details that happened earlier in the story.\n"

    for plot_point in plot_details:
        snippets += f"- {plot_point}\n"

        snippets = snippets.strip()
        snippets += "\n"

    message = (
        "\n\nUse the following context as information about the story, inside <context></context> XML tags.\n\n"
        f"<context>\n{snippets}</context>\n"
        "When answering to user:\n"
        "- Use the context to enhance your knowledge of the story.\n"
        "- If you don't know, do not ask for clarification.\n"
        "Do not mention that you obtained the information from the context.\n"
        "Do not mention the context.\n"
        f"Continue the story according to the user's directions."
    )

    return message


def write_log(text):
    with open(f"/tmp/test-memories", "a") as file:
        file.write(text + "\n")


def split_messages(messages, keep_amount):
    if len(messages) <= keep_amount:
        return messages[:], []

    recent_messages = messages[-keep_amount:]
    old_messages = messages[:-keep_amount]
    return recent_messages, old_messages


def chunk_messages(messages, chunk_size):
    return [messages[i:i + chunk_size] for i in range(0, len(messages), chunk_size)]

def llm_messages_to_user_messages(messages):
    return [
        {'role': 'user', 'content': msg['content']}
        for msg in messages if msg['role'] == 'assistant'
    ]

# Das Filter
class Filter:
    class Valves(BaseModel):
        def summarizer_model(self, body):
            if self.summarizer_model_id == "":
                # This will be the model ID in the convo. If not base
                # model, it will cause problems.
                return body["model"]
            else:
                return self.summarizer_model_id

        summarizer_model_id: str = Field(
            default="",
            description="Model used to summarize the conversation. Must be a base model.",
        )

        n_last_messages: int = Field(
            default=4, description="Number of last messages to retain."
        )
        pass



    class UserValves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    def extract_convo_id(self, messages):
        """Extract ID of first message to use as conversation ID."""
        if len(messages) > 0:
            first_user_message = next(
                (message for message in messages if message.get("role") == "user"), None
            )

            if first_user_message and 'id' in first_user_message:
                return first_user_message['id']
            else:
                raise ValueError("No messages found to extract conversation ID")
        else:
            raise ValueError("No messages found to extract conversation ID")


    async def summarize(self, messages) -> Optional[SummarizerResponse]:
        message_to_summarize = get_last_assistant_message(messages)
        if message_to_summarize:
            summarizer = Summarizer(model=self.summarizer_model_id, message=message_to_summarize)
            return await summarizer.summarize()
        else:
            return None

    async def send_outlet_status(self, event_emitter, done: bool):
        description = (
            "Analyzing Narrative (do not reply until this is done)" if not done else
            "Narrative analysis complete (you may now reply)."
        )
        await event_emitter({
            "type": "status",
            "data": {
                "description": description,
                "done": done,
            },
        })

    async def set_enriching_status(self, state: str):
        if not self.event_emitter:
            return

        done = state == "done"
        description = "Enriching Narrative"

        if state == "init": description = f"{description}: Initializing..."
        if state == "searching": description = f"{description}: Searching..."
        if state == "analyzing": description = f"{description}: Analyzing..."

        description = (
            description if not done else
            "Enrichment Complete"
        )

        await self.event_emitter({
            "type": "status",
            "data": {
                "description": description,
                "done": done,
            },
        })

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict],
        __event_emitter__: Callable[[Any], Awaitable[None]],
    ) -> dict:
        # Useful things to have around.
        self.event_emitter = __event_emitter__
        self.summarizer_model_id = self.valves.summarizer_model(body)

        await self.send_outlet_status(__event_emitter__, False)
        messages = body['messages']
        convo_id = self.extract_convo_id(messages)

        # summarize into plot points.
        summary = await self.summarize(messages)
        story = Story(
            convo_id=convo_id, client=CHROMA_CLIENT,
            embedding_func=EMBEDDING_FUNCTION,
            messages=messages
        )

        story.switch_convo()

        if summary:
            story.embed_summary(summary)

        await self.send_outlet_status(__event_emitter__, True)
        return body

    async def generate_enrichment_queries(self, messages) -> SummarizerResponse:
        last_response = get_last_assistant_message(messages)
        user_input = get_last_user_message(messages)

        query_message = ""
        if last_response: query_message += f"## Assistant\n\n{last_response}\n\n"
        if user_input: query_message += f"## User\n\n{user_input}\n\n"
        query_message = query_message.strip()

        summarizer = Summarizer(
            model=self.summarizer_model_id,
            message=query_message,
            prompt=QUERY_PROMPT
        )

        return await summarizer.summarize()

    async def summarize_enrichment(
            self,
            messages,
            character_results: List[ChromaDocument],
            plot_results: List[ChromaDocument]
    ) -> SummarizerResponse:
        last_response = get_last_assistant_message(messages)
        user_input = get_last_user_message(messages)

        character_details = [r['doc'] for r in character_results]
        plot_details = [r['doc'] for r in plot_results]

        narrative_message = ""
        if last_response: narrative_message += f"## Assistant\n\n{last_response}\n\n"
        if user_input: narrative_message += f"## User\n\n{user_input}\n\n"
        narrative_message = narrative_message.strip()

        summarization_prompt = create_enrichment_summary_prompt(
            narrative=narrative_message,
            plot_details=plot_details,
            character_details=character_details
        )

        summarizer = Summarizer(
            model=self.summarizer_model_id,
            message=narrative_message,
            prompt=summarization_prompt
        )

        return await summarizer.summarize()


    async def enrich(self, story: Story, messages) -> SummarizerResponse:
        await self.set_enriching_status("searching")
        query_generation_result = await self.generate_enrichment_queries(messages)
        character_results = [result
                             for query in query_generation_result.characters
                             for result in story.query_characters(query)]

        plot_results = [result
                             for query in query_generation_result.plot
                             for result in story.query_plot(query)]

        await self.set_enriching_status("analyzing")
        return await self.summarize_enrichment(messages, character_results, plot_results)


    async def update_system_message(self, messages, system_message):
        story = Story(
            convo_id=None, client=CHROMA_CLIENT,
            embedding_func=EMBEDDING_FUNCTION,
            messages=messages
        )

        story.switch_convo()

        if story.convo_id == "<unset>":
            return

        enrichment_summary: SummarizerResponse = await self.enrich(story, messages)
        context = create_context(enrichment_summary)

        if context:
            system_message["content"] += context


    async def inlet(
            self,
            body: dict,
            __user__: Optional[dict],
            __event_emitter__: Callable[[Any], Awaitable[None]]
    ) -> dict:
        # Useful properties to have around.
        self.event_emitter = __event_emitter__
        self.summarizer_model_id = self.valves.summarizer_model(body)
        await self.set_enriching_status("init")
        messages = body["messages"]

        # Ensure we always keep the system prompt
        system_prompt = next(
            (message for message in messages if message.get("role") == "system"), None
        )

        if system_prompt:
            all_messages = [
                message for message in messages if message.get("role") != "system"
            ]

            recent_messages, old_messages = split_messages(all_messages, self.valves.n_last_messages)
            most_recent_messages = messages[-self.valves.n_last_messages :]
        else:
            system_prompt = { "id": str(uuid.uuid4()), "role": "system", "content": "" }
            recent_messages, old_messages = split_messages(messages, self.valves.n_last_messages)

        await self.update_system_message(messages, system_prompt)
        recent_messages.insert(0, system_prompt)

        body["messages"] = recent_messages
        await self.set_enriching_status("done")
        return body
