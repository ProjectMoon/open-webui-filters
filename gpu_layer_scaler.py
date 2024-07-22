"""
title: GPU Scaling Filter
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 0.1.0
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
    if "info" in model:
        model_info = model["info"]
        return model_info["base_model_id"] if "base_model_id" in model_info else model["id"]
    else:
        return None

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

class Filter:
    class Valves(BaseModel):
        reduction_start: int = Field(
            default=20, description="Amount of GPU layers to reduce to immediately on failure"
        )
        scaling_step: int = Field(
            default=5, description="Amount of GPU layers to reduce by on continued failures"
        )
        show_status: bool = Field(
            default=True, description="Show status message when running downscaled model."
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    async def send_message_adjusting(self, done: bool, amount: int=0, steps: int=0):
        if steps > 0:
            steps_desc = f"reduced by {steps}"
        else:
            steps_desc = "initial reduction"

        desc = (
            "Downscaling GPU layers..." if not done
            else f"GPU layers downscaled to {amount} ({steps_desc}). Please retry.")

        await self.event_emitter(
            {
                "type": "status",
                "data": {
                    "description": desc,
                    "done": done
                },
            }
        )

    async def send_message_downscaled(self):
        await self.event_emitter(
            {
                "type": "status",
                "data": {
                    "description": "Running at reduced GPU capacity. Responses will be slower.",
                    "done": True
                },
            }
        )

    def get_num_layers_for_model(
            self,
            gpu_layer_info: GpuChatState,
            __model__: dict
    ) -> Optional[int]:
        model_id = extract_model_id(__model__)
        if model_id:
            return gpu_layer_info.get_gpu_layers_for_model(model_id)
        else:
            return None

    async def downscale(self, model):
        """Update tracked downscale GPU layers for this chat + model."""
        # this logic is currently very basic. does not yet take into
        # account the actual number of layers in a model. but it's
        # better than nothing. if this is the first failure (no entry
        # in gpu chat state), set number of layers to the valve
        # parameter. if this is a subsequent failure (we have entry
        # for this chat already), reduce by the step valve parameter,
        # to a minimum of CPU (100% cpu).
        await self.send_message_adjusting(False)
        gpu_layer_info = GpuChatState(CHROMA_CLIENT, self.session_info.chat_id)
        num_layers = self.get_num_layers_for_model(gpu_layer_info, model)
        print(f"num layers is {num_layers}")
        downscale_steps = 0

        if num_layers:
            print(f"Downscaling layers by {self.valves.scaling_step}")
            num_layers -= self.valves.scaling_step
            downscale_steps = self.valves.scaling_step
            if num_layers < 0:
                num_layers = 0
        else:
            num_layers = self.valves.reduction_start

        model_id = extract_model_id(model)
        if model_id:
            gpu_layer_info.set_gpu_layers(model_id, num_layers)
            await self.send_message_adjusting(True, amount=num_layers, steps=downscale_steps)
            print(
                f"Set GPU layers for chat {self.session_info.chat_id} to {num_layers}"
            )

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
    ) -> dict:
        """Intercept incoming messages and downscale if necessary."""
        self.event_emitter = __event_emitter__
        self.session_info = extract_session_info(__event_emitter__)

        if self.session_info and __model__:
            model_id = extract_model_id(__model__)
            gpu_layer_info = GpuChatState(CHROMA_CLIENT, self.session_info.chat_id)
            num_layers = self.get_num_layers_for_model(gpu_layer_info, __model__)

            if num_layers and "options" in body:
                body["options"]["num_gpu"] = num_layers
                if self.valves.show_status:
                    await self.send_message_downscaled()
                print(f"Downscaled GPU layers for incoming request for {model_id} to {num_layers}")

        return body

    async def outlet(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
    ) -> dict:
        """On response failure, downscale the GPU layers for next try."""
        self.event_emitter = __event_emitter__
        self.session_info = extract_session_info(__event_emitter__)

        if not self.session_info or not __model__:
            return body

        if len(body["messages"]) == 0:
            return body

        last_reply = body["messages"][-1]
        broke = last_reply["content"] == "" and last_reply["info"] == {}

        if broke:
            # while we could actually redo the message itself, it is
            # useless, because open web ui does not currently have a
            # way to clear error state when message content is
            # replaced. so we just lower gpu layers and tell user to
            # try again. the inlet will intercept the incoming request
            # and lower the gpu layers.
            await self.downscale(__model__)

        return body
