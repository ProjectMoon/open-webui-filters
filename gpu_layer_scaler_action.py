"""
title: GPU Scaling Filter Actions
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 0.2.0
license: AGPL-3.0+
required_open_webui_version: 0.3.9
"""

# Documentation: https://git.agnos.is/projectmoon/open-webui-filters

# System Imports
import chromadb
from chromadb import ClientAPI as ChromaAPI
from chromadb import Collection as ChromaCollection
from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional, Literal
import json

# OpenWebUI imports
from config import CHROMA_CLIENT
from utils.misc import get_last_user_message, get_last_assistant_message
from apps.ollama.main import generate_chat_completion, GenerateChatCompletionForm
from apps.webui.models.users import UserModel

class GpuChatState:
    """
    Get or set GPU layer count by base model for a given chat.
    """

    collection_name = "gpu_layers_by_chat"

    def __init__(self, chroma_client: ChromaAPI, chat_id: str):
        self.chroma_client = chroma_client
        self.chat_id = chat_id
        self.gpu_layers = {}

    def _get_collection(self) -> ChromaCollection:
        return self.chroma_client.get_or_create_collection(
            name=GpuChatState.collection_name
        )

    def _parse_results(self, results) -> dict:
        if 'documents' in results:
            doc = results['documents'][0] if len(results['documents']) > 0 else None
            return json.loads(doc) if doc else {}
        else:
            return {}

    def get_gpu_layers(self):
        coll = self._get_collection()

        if self.gpu_layers == {}:
            self.gpu_layers = self._parse_results(
                coll.get(ids=[self.chat_id], include=["documents"])
            )

        return self.gpu_layers

    def get_gpu_layers_for_model(self, model_id: str) -> Optional[int]:
        info = self.get_gpu_layers()
        return info[model_id] if model_id in info else None

    def set_gpu_layers(self, model: str, amount: int):
        # set gpu layers for this chat.
        self.gpu_layers[model] = amount
        self._get_collection().upsert(
            ids=[self.chat_id],
            documents=[json.dumps(self.gpu_layers)]
        )
        self.gpu_layers = self.get_gpu_layers()

    def del_gpu_layers(self, model: str):
        self.gpu_layers = self.get_gpu_layers()
        if model in self.gpu_layers:
            del self.gpu_layers[model]
            self._get_collection().upsert(
                ids=[self.chat_id],
                documents=[json.dumps(self.gpu_layers)]
            )


class SessionInfo(BaseModel):
    chat_id: str
    message_id: str
    session_id: str

def dict_to_attributes(input_dict):
    class AttrDict:
        def __init__(self, attr_dict):
            for key, value in attr_dict.items():
                setattr(self, key, value)

    return AttrDict(input_dict)

def extract_model_id(model: dict) -> Optional[str]:
    model_id = None

    if "info" in model:
        if "base_model_id" in model["info"]:
            model_id = model["info"]["base_model_id"]
    else:
        if "ollama" in model and "id" in model["ollama"]:
            model_id = model["ollama"]["id"]

    if not model_id:
        model_id = model["id"]

    return model_id

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

class Action:
    class Valves(BaseModel):
        # reduction_start: int = Field(
        #     default=20, description="Amount of GPU layers to reduce to immediately on failure"
        # )
        # scaling_step: int = Field(
        #     default=5, description="Amount of GPU layers to reduce by on continued failures"
        # )
        # show_status: bool = Field(
        #     default=True, description="Show status message when running downscaled model."
        # )
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass


    async def send_message_reset(self):
        await self.event_emitter(
            {
                "type": "status",
                "data": {
                    "description": "Reset GPU downscaling.",
                    "done": True
                },
            }
        )

    async def reset(self, model):
        """Remove tracked downscale GPU layers for this chat + model."""
        model_id = extract_model_id(model)

        if not model_id:
            print("Could not extract model ID for GPU downscaling reset!")
            return

        gpu_layer_info = GpuChatState(CHROMA_CLIENT, self.session_info.chat_id)
        gpu_layer_info.del_gpu_layers(model_id)
        await self.send_message_reset()
        print(
            f"Reset GPU layers in {self.session_info.chat_id} for {model_id}"
        )

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
        __model__=None
    ) -> Optional[dict]:
        """Reset GPU layers."""
        if not __model__ or __model__["owned_by"] != "ollama":
            return

        self.event_emitter = __event_emitter__
        self.session_info = extract_session_info(__event_emitter__)

        if not self.session_info or not __model__:
            return

        await self.reset(__model__)
