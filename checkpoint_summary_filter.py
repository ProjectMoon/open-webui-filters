"""
title: Checkpoint Summary Filter
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 0.1.0
license: AGPL-3.0+
required_open_webui_version: 0.3.9
"""

# Documentation: https://git.agnos.is/projectmoon/open-webui-filters

# System imports
import asyncio
import hashlib
import uuid
import json
import re
import logging

from typing import Optional, List, Dict, Callable, Any, NewType, Tuple, Awaitable, ClassVar
from typing_extensions import TypedDict, NotRequired
from collections import deque

# Libraries available to OpenWebUI
from pydantic import BaseModel as PydanticBaseModel, Field
import chromadb
from chromadb import Collection as ChromaCollection
from chromadb.api.types import Document as ChromaDocument

# OpenWebUI imports
from config import CHROMA_CLIENT
from apps.rag.main import app as rag_app
from apps.ollama.main import app as ollama_app
from apps.ollama.main import show_model_info, ModelNameForm
from utils.misc import get_last_user_message, get_last_assistant_message
from main import generate_chat_completions

from apps.webui.models.chats import Chats
from apps.webui.models.models import Models
from apps.webui.models.users import Users

# Embedding (not yet used)
EMBEDDING_FUNCTION = rag_app.state.EMBEDDING_FUNCTION
EmbeddingFunc = NewType('EmbeddingFunc', Callable[[str], List[Any]])

# Prompts
SUMMARIZER_PROMPT = """
### Main Instructions

You are a chat conversation summarizer. Your task is to summarize the given
portion of an ongoing conversation. First, determine if the conversation is
a regular chat between the user and the assistant, or if the conversation is
part of a story or role-playing session.

Summarize the important parts of the given chat between the user and the
assistant. Limit your summary to one paragraph. Make sure your summary is
detailed. Write the summary as if you are summarizing part of a larger
conversation.

### Regular Chat

If the conversation is a regular chat, write your summary referring to the
ongoing conversation as a chat. Refer to the user and the assistant as user
and assistant. Do not refer to yourself as the assistant.

### Story or Role-Playing Session

If the conversation is a story or role-playing session, write your summary
referring to the conversation as an ongoing story. Do not refer to the user
or assistant in your summary. Only use the names of the characters, places,
and events in the story.
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
    summary: str


class Summarizer(BaseModel):
    messages: List[dict]
    model: str
    prompt: str = SUMMARIZER_PROMPT

    async def summarize(self) -> Optional[SummarizerResponse]:
        sys_message: Message = { "role": "system", "content": SUMMARIZER_PROMPT }
        user_message: Message = {
            "role": "user",
            "content": "Make a detailed summary of everything up to this point."
        }

        messages = [sys_message] + self.messages + [user_message]

        request = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10s"
        }

        resp = await generate_chat_completions(request)
        if "choices" in resp and len(resp["choices"]) > 0:
            content: str = resp["choices"][0]["message"]["content"]
            return SummarizerResponse(summary=content)
        else:
            return None


class Checkpoint(BaseModel):
    # chat id
    chat_id: str

    # the message ID this checkpoint was created from.
    message_id: str

    # index of the message in the message input array. in the inlet
    # function, we do not have access to incoming message ids for some
    # reason. used as a fallback to drop old context when
    message_index: int = 0

    # the "slug", or chain of messages, that led to this point.
    slug: str

    # actual summary of messages.
    summary: str

    # if we try to put a type hint on this, it gets mad.
    @staticmethod
    def from_json(obj: dict):
        try:
            return Checkpoint(
                chat_id=obj["chat_id"],
                message_id=obj["message_id"],
                message_index=obj["message_index"],
                slug=obj["slug"],
                summary=obj["summary"]
            )
        except:
            return None

    def to_json(self) -> str:
        return self.model_dump_json()


class Checkpointer(BaseModel):
    """Manages summary checkpoints in a single chat."""
    chat_id: str
    summarizer_model: str = ""
    chroma_client: chromadb.ClientAPI
    messages: List[dict]=[] # stripped set of messages
    full_messages: List[dict]=[] # all the messages
    embedding_func: EmbeddingFunc=(lambda a: 0)

    collection_name: ClassVar[str] = "chat_checkpoints"

    def _get_collection(self) -> ChromaCollection:
        return self.chroma_client.get_or_create_collection(
            name=Checkpointer.collection_name
        )


    def _insert_checkpoint(self, checkpoint: Checkpoint):
        coll = self._get_collection()
        checkpoint_doc = checkpoint.to_json()
        # Insert the checkpoint itself with slug as ID.
        coll.upsert(
            ids=[checkpoint.slug],
            documents=[checkpoint_doc],
            metadatas=[{ "chat_id": self.chat_id, "type": "checkpoint" }],
            embeddings=[self.embedding_func(checkpoint_doc)]
        )

        # Update the chat info doc for this chat.
        coll.upsert(
            ids=[self.chat_id],
            documents=[json.dumps({ "current_checkpoint": checkpoint.slug })],
            embeddings=[self.embedding_func(self.chat_id)]
        )

    def _calculate_slug(self) -> Optional[str]:
        if len(self.messages) == 0:
            return None

        message_ids = [msg["id"] for msg in reversed(self.messages)]
        slug = "|".join(message_ids)
        return hashlib.sha256(slug.encode()).hexdigest()

    def _get_state(self):
        resp = self._get_collection().get(ids=[self.chat_id], include=["documents"])
        state: dict = (json.loads(resp["documents"][0])
                 if resp["documents"] and len(resp["documents"]) > 0
                 else { "current_checkpoint": None })
        return state


    def _find_message_index(self, message_id: str) -> Optional[int]:
        for idx, message in enumerate(self.full_messages):
            if message["id"] == message_id:
                return idx
        return None

    def nuke_checkpoints(self):
        """Delete all checkpoints for this chat."""
        coll = self._get_collection()

        checkpoints = coll.get(
            include=["documents"],
            where={"chat_id": self.chat_id}
        )

        self._get_collection().delete(
            ids=[self.chat_id] + checkpoints["ids"]
        )

    async def create_checkpoint(self) -> str:
        summarizer = Summarizer(model=self.summarizer_model, messages=self.messages)
        resp = await summarizer.summarize()
        if resp:
            slug = self._calculate_slug()
            checkpoint_message = self.messages[-1]
            checkpoint_index = self._find_message_index(checkpoint_message["id"])

            checkpoint = Checkpoint(
                chat_id = self.chat_id,
                slug = self._calculate_slug(),
                message_id = checkpoint_message["id"],
                message_index = checkpoint_index,
                summary = resp.summary
            )

            self._insert_checkpoint(checkpoint)
            return slug

    def get_checkpoint(self, slug: Optional[str]) -> Optional[Checkpoint]:
        if not slug:
            return None

        resp = self._get_collection().get(ids=[slug], include=["documents"])
        checkpoint = (resp["documents"][0]
                      if resp["documents"] and len(resp["documents"]) > 0
                      else None)

        if checkpoint:
            return Checkpoint.from_json(json.loads(checkpoint))
        else:
            return None

    def get_current_checkpoint(self) -> Optional[Checkpoint]:
        state = self._get_state()
        return self.get_checkpoint(state["current_checkpoint"])


#########################
# Utilities
#########################

class SessionInfo(BaseModel):
    chat_id: str
    message_id: str
    session_id: str


def extract_session_info(event_emitter) -> Optional[SessionInfo]:
    """The latest innovation in hacky workarounds."""
    try:
        info = event_emitter.__closure__[0].cell_contents
        return SessionInfo(
            chat_id=info["chat_id"],
            message_id=info["message_id"],
            session_id=info["session_id"]
        )
    except:
        return None


def predicted_token_use(messages) -> int:
    """Parse most recent message to calculate estimated token use."""
    if len(self.messages == 0):
        return 0

    # Naive assumptions:
    #  - 1 word = 1 token.
    #  - 1 period, comma, or colon = 1 token
    message = messages[-1]
    return len(list(filter(None, re.split(r"\s|(;)|(,)|(\.)|(:)|\n", message))))

def is_big_convo(messages, num_ctx: int=8192) -> bool:
    """
    Attempt to detect large pre-existing conversation by looking at
    recent eval counts from messages and comparing against given
    num_ctx. We check all messages for an eval count that goes above
    the context limit. It doesn't matter where in the message list; if
    it's somewhere in the middle, it means that there was a context
    shift.
    """
    for message in messages:
        if "info" in message:
            tokens_used = (message["info"]["eval_count"] +
                           message["info"]["prompt_eval_count"])
        else:
            tokens_used = 0

        if tokens_used >= num_ctx:
            return True

    return False


def hit_context_limit(
        messages,
        num_ctx: int=8192,
        wiggle_room: int=1000
) -> Tuple[bool, int]:
    """
    Determine if we've hit the context limit, within some reasonable
    estimation. We have a defined 'wiggle room' that is subtracted
    from the num_ctx parameter, in order to capture near-filled
    contexts. We do it this way because we're summarizing on output,
    rather than before input (inlet function doesn't have enough
    info).
    """
    if len(messages) == 0:
        return False, 0

    last_message = messages[-1]
    tokens_used = 0
    if "info" in last_message:
        tokens_used = (last_message["info"]["eval_count"] +
                       last_message["info"]["prompt_eval_count"])

    if tokens_used >= (num_ctx - wiggle_room):
        amount_over = tokens_used - num_ctx
        amount_over = 0 if amount_over < 0 else amount_over
        return True, amount_over
    else:
        return False, 0

def extract_base_model_id(model: dict) -> Optional[str]:
    if "base_model_id" not in model["info"]:
        return None

    base_model_id = model["info"]["base_model_id"]
    if not base_model_id:
        base_model_id = model["id"]

    return base_model_id

def extract_owu_model_param(model_obj: dict, param_name: str):
    """
    Extract a parameter value from the DB definition of a model
    that is based on another model.
    """
    if not "params" in model_obj["info"]:
        return None

    params = model_obj["info"]["params"]
    return params.get(param_name, None)

def extract_owu_base_model_param(base_model_id: str, param_name: str):
    """Extract a parameter value from the DB definition of an ollama base model."""
    base_model = Models.get_model_by_id(base_model_id)

    if not base_model:
        return None

    base_model.params = base_model.params.model_dump()
    return base_model.params.get(param_name, None)

def extract_ollama_response_param(model: dict, param_name: str):
    """Extract a parameter value from ollama show API response."""
    if "parameters" not in model:
        return None

    for line in model["parameters"].splitlines():
        if line.startswith(param_name):
            return line.lstrip(param_name).strip()

    return None

async def get_model_from_ollama(model_id: str, user_id) -> Optional[dict]:
    """Call ollama show API and return model information."""
    curr_user = Users.get_user_by_id(user_id)
    try:
        return await show_model_info(ModelNameForm(name=model_id), user=curr_user)
    except Exception as e:
        print(f"Could not get model info: {e}")
        return None

async def calculate_num_ctx(chat_id: str, user_id, model: dict) -> int:
    """
    Attempt to discover the current num_ctx parameter in many
    different ways.
    """
    # first check the open-webui chat parameters.
    chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
    if chat:
        # this might look odd, but the chat field is a json blob of
        # useful info.
        chat = json.loads(chat.chat)
        if "params" in chat and "num_ctx" in chat["params"]:
            return chat["params"]["num_ctx"]

    # then check open web ui model def
    num_ctx = extract_owu_model_param(model, "num_ctx")
    if num_ctx:
        return num_ctx

    # then check open web ui base model def.
    base_model_id = extract_base_model_id(model)
    if not base_model_id:
        # fall back to default in case of weirdness.
        return 2048

    num_ctx = extract_owu_base_model_param(base_model_id, "num_ctx")
    if num_ctx:
        return num_ctx

    # THEN check ollama directly.
    base_model = await get_model_from_ollama(base_model_id, user_id)
    num_ctx = extract_ollama_response_param(base_model, "num_ctx")
    if num_ctx:
        return num_ctx

    # finally, return default.
    return 2048



class Filter:
    class Valves(BaseModel):
        def summarizer_model(self, body):
            if self.summarizer_model_id == "":
                return extract_base_model_id(body["model"])
            else:
                return self.summarizer_model_id

        summarize_large_contexts: bool = Field(
            default=False,
            description=(
                f"Whether or not to use a large context model to summarize large "
                f"pre-existing conversations."
            )
        )
        wiggle_room: int = Field(
            default=1000,
            description=(
                "Amount of token 'wiggle room' for estimating when a context shift occurs. "
                "Subtracted from num_ctx when checking if summarization is needed."
            )
        )
        summarizer_model_id: str = Field(
            default="",
            description="Model used to summarize the conversation. Must be a base model.",
        )
        large_summarizer_model_id: str = Field(
            default="",
            description=(
                "Model used to summarize large pre-existing contexts. "
                "Must be a base model with a context size large enough "
                "to fit the conversation."
            )
        )
        pass

    class UserValves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass


    def load_current_chat(self) -> dict:
        # the chat property of the model is the json blob that holds
        # all the interesting stuff
        chat = (Chats
                .get_chat_by_id_and_user_id(self.session_info.chat_id, self.user["id"])
                .chat)

        return json.loads(chat)

    def get_messages_for_checkpointing(self, messages, num_ctx, last_checkpointed_id):
        """
        Assemble list of messages to checkpoint, based on current
        state and valve settings.
        """
        message_chain = deque()
        for message in reversed(messages):
            if message["id"] == last_checkpointed_id:
                break
            message_chain.appendleft(message)

        message_chain = list(message_chain) # the lazy way

        # now we check if we are a big conversation, and if valve
        # settings allow that kind of summarization.
        summarizer_model = self.valves.summarizer_model
        if is_big_convo(messages, num_ctx) and not self.valves.summarize_large_contexts:
            # must summarize using small model. for now, drop to last
            # N messages.
            print((
                "Dropping all but last 4 messages to summarize "
                "large convo without large model."
            ))
            message_chain = message_chain[-4:]

        return message_chain


    async def create_checkpoint(
            self,
            messages: List[dict],
            last_checkpointed_id: Optional[str]=None,
            num_ctx: int=8192
    ):
        if len(messages) == 0:
            return

        print(f"[{self.session_info.chat_id}] Detected context shift. Summarizing.")
        await self.set_summarizing_status(done=False)
        last_message = messages[-1] # should check for role = assistant
        curr_message_id: Optional[str] = (
            last_message["id"] if last_message else None
        )

        if not curr_message_id:
            return

        # strip messages down to what is in the current checkpoint.
        message_chain = self.get_messages_for_checkpointing(
            messages, num_ctx, last_checkpointed_id
        )

        # we should now have a list of messages that is just within
        # the current context limit.
        summarizer_model = self.valves.summarizer_model_id
        if is_big_convo(message_chain, num_ctx) and self.valves.summarize_large_contexts:
            print("Summarizing LARGE context!")
            summarizer_model = self.valves.large_summarizer_model_id


        checkpointer = Checkpointer(
            chat_id=self.session_info.chat_id,
            summarizer_model=summarizer_model,
            chroma_client=CHROMA_CLIENT,
            full_messages=messages,
            messages=message_chain
        )

        try:
            slug = await checkpointer.create_checkpoint()
            await self.set_summarizing_status(done=True)
            print(("Summarization checkpoint created in chat "
                   f"'{self.session_info.chat_id}': {slug}"))
        except Exception as e:
            print(f"Error creating summary: {str(e)}")
            await self.set_summarizing_status(
                done=True, message=f"Error summarizing: {str(e)}"
            )


    def update_chat_with_checkpoint(self, messages: List[dict], checkpoint: Checkpoint):
        if len(messages) < checkpoint.message_index:
            # do not mess with anything if the index doesn't even
            # exist anymore. need a new checkpoint.
            return messages

        # proceed with altering the system prompt. keep system prompt,
        # if it's there, and add summary to it. summary will become
        # system prompt if there is no system prompt.
        convo_messages = [
            message for message in messages if message.get("role") != "system"
        ]

        system_prompt = next(
            (message for message in messages if message.get("role") == "system"), None
        )

        summary_message = f"Summary of conversation so far:\n\n{checkpoint.summary}"

        if system_prompt:
            system_prompt["content"] += f"\n\n{summary_message}"
        else:
            system_prompt = { "role": "system", "content": summary_message }


        # drop old messages, reapply system prompt.
        messages = self.apply_checkpoint(checkpoint, messages)
        return [system_prompt] + messages


    async def send_message(self, message: str):
        await self.event_emitter({
            "type": "status",
            "data": {
                "description": message,
                "done": True,
            },
        })

    async def set_summarizing_status(self, done: bool, message: Optional[str]=None):
        if not self.event_emitter:
            return

        if not done:
            description = (
                "Summarizing conversation due to reaching context limit (do not reply yet)."
            )
        else:
            description = (
                "Summarization complete (you may now reply)."
            )

        if message:
            description = message

        await self.event_emitter({
            "type": "status",
            "data": {
                "description": description,
                "done": done,
            },
        })

    def apply_checkpoint(
            self, checkpoint: Checkpoint, messages: List[dict]
    ) -> List[dict]:
        """
        Possibly shorten the message context based on a checkpoint.
        This works two ways: if the messages have IDs (outlet
        filter), split by message ID (very reliable). Otherwise,
        attempt to split by on the recorded message index (inlet
        filter; not very reliable).
        """

        # first attempt to drop everything before the checkpointed
        # message id.
        split_point = 0
        for idx, message in enumerate(messages):
            if "id" in message and message["id"] == checkpoint.message_id:
                split_point = idx
                break

        # if we can't find the ID to split on, fall back to message
        # index if possible. this can happen during message
        # regeneration, for example. or if we're called from the inlet
        # filter, which doesn't have access to message ids.
        if split_point == 0 and checkpoint.message_index <= len(messages):
            split_point = checkpoint.message_index

        orig = len(messages)
        messages = messages[split_point:]
        print((f"[{self.session_info.chat_id}] Dropped context to {len(messages)} "
               f"messages (from {orig})"))
        return messages


    async def handle_nuke(self, body):
        checkpointer = Checkpointer(
            chat_id=self.session_info.chat_id,
            chroma_client=CHROMA_CLIENT
        )
        checkpointer.nuke_checkpoints()
        await self.send_message("Deleted all checkpoint for chat.")

        body["messages"][-1]["content"] = (
            "Respond ony with: 'Deleted all checkpoint for chat.'"
        )

        body["messages"] = body["messages"][-1:]
        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict],
        __model__: Optional[dict],
        __event_emitter__: Callable[[Any], Awaitable[None]],
    ) -> dict:
        # Useful things to have around.
        self.user = __user__
        self.model = __model__
        self.session_info = extract_session_info(__event_emitter__)
        self.event_emitter = __event_emitter__
        self.summarizer_model_id = self.valves.summarizer_model(body)

        # global filters apply to requests coming in through proxied
        # API. If we're not an OpenWebUI chat, abort mission.
        if not self.session_info:
            return body

        if not self.model or self.modle["owned_by"] != "ollama":
            return body

        messages = body["messages"]

        num_ctx = await calculate_num_ctx(
            chat_id=self.session_info.chat_id,
            user_id=self.user["id"],
            model=self.model
        )

        # apply current checkpoint ONLY for purposes of calculating if
        # we have hit num_ctx within current checkpoint.
        checkpointer = Checkpointer(
            chat_id=self.session_info.chat_id,
            chroma_client=CHROMA_CLIENT
        )

        checkpoint = checkpointer.get_current_checkpoint()
        messages_for_ctx_check = (self.apply_checkpoint(checkpoint, messages)
                                  if checkpoint else messages)

        hit_limit, amount_over = hit_context_limit(
            messages=messages_for_ctx_check,
            num_ctx=num_ctx,
            wiggle_room=self.valves.wiggle_room
        )

        if hit_limit:
            # we need the FULL message list to do proper summarizing,
            # because we might be summarizing a hug context.
            await self.create_checkpoint(
                messages=messages,
                num_ctx=num_ctx,
                last_checkpointed_id=checkpoint.message_id if checkpoint else None
            )

        print(f"[{self.session_info.chat_id}] Done checking for summarization")
        return body


    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict],
        __model__: Optional[dict],
        __event_emitter__: Callable[[Any], Awaitable[None]]
    ) -> dict:
        # Useful properties to have around.
        self.user = __user__
        self.model = __model__
        self.session_info = extract_session_info(__event_emitter__)
        self.event_emitter = __event_emitter__
        self.summarizer_model_id = self.valves.summarizer_model(body)

        # global filters apply to requests coming in through proxied
        # API. If we're not an OpenWebUI chat, abort mission.
        if not self.session_info:
            return body

        if not self.model or self.modle["owned_by"] != "ollama":
            return body

        # super basic external command handling (delete checkpoints).
        user_msg = get_last_user_message(body["messages"])
        if user_msg and user_msg == "!nuke":
            return await self.handle_nuke(body)

        # apply current checkpoint to the chat: adds most recent
        # summary to system prompt, and drops all messages before the
        # checkpoint.
        checkpointer = Checkpointer(
            chat_id=self.session_info.chat_id,
            chroma_client=CHROMA_CLIENT
        )

        checkpoint = checkpointer.get_current_checkpoint()
        if checkpoint:
            print((
                f"Using checkpoint {checkpoint.slug} for "
                f"conversation {self.session_info.chat_id}"
            ))

            body["messages"] = self.update_chat_with_checkpoint(body["messages"], checkpoint)

        return body
