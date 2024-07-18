"""
title: GPU scaling router
author: open-webui, atgehrhardt
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.1.4
required_open_webui_version: 0.3.8
"""

from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional, Literal
import json

from utils.misc import get_last_user_message, get_last_assistant_message
from apps.ollama.main import generate_chat_completion, GenerateChatCompletionForm
from apps.webui.models.users import UserModel

# To get ROCm VRAM use: rocm-smi --showmeminfo vram --json
# To figure out GPU layers in use: janky ass bullshit!
#  1. Use ollama API to get modelfile from model info.
#  2. Pull actual file path of model out of the modelfile.
#  3. Scan running processes for the one that is using our file.
#  4. Parse its command line to get number of GPU layers.

# How to stabilize VRAM use: we don't want to change layers all the
# time, because it'll cause the model to reload a lot.
# We need to maintain state per convo (yay). Shove it into ChromaDB!

# Could also try summing up tokens? Or calculating vram use of model
# vs vram use of rocm, and do nothing if below %


def write_log(text):
    with open(f"/tmp/test-memories", "a") as file:
        file.write(text + "\n")

def dict_to_attributes(input_dict):
    class AttrDict:
        def __init__(self, attr_dict):
            for key, value in attr_dict.items():
                setattr(self, key, value)

    return AttrDict(input_dict)

def convert_user(user):
    user['info'] = {}
    return dict_to_attributes(user)

class Filter:
    class Valves(BaseModel):
        scaling_start: int = Field(
            default=90,
            description="VRAM usage percent to start scaling back GPU layers",
        )
        scaling_step: int = Field(
            default=3, description="Amount of GPU layers to reduce"
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    async def message_adjusting(self, done: bool):
        await self.event_emitter(
            {
                "type": "status",
                "data": {
                    "description": "Adjusting GPU layers",
                    "done": done,
                },
            }
        )

    async def retry_message(self, body, user):
        request = GenerateChatCompletionForm(
            model=body["model"],
            messages=body["messages"],
            stream=False,
            keep_alive="10s",
            options={"num_gpu": 1},
        )

        return await generate_chat_completion(request, user=user)

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
    ) -> dict:
        self.event_emitter = __event_emitter__
        return body

    async def outlet(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
    ) -> dict:
        user = convert_user(__user__)
        self.event_emitter = __event_emitter__
        if len(body["messages"]) == 0:
            return body

        message = body["messages"][-1]
        write_log("got a message")
        write_log(f"message: {str(message)}")

        broke = message["content"] == "" and message["info"] == {}
        if broke:
            # at this point, we COULD set status and attempt to reduce
            # the GPU layers?
            await self.message_adjusting(False)
            del body["messages"][-1]
            retried = await self.retry_message(body, user)
            await self.message_adjusting(True)
            message["content"] = get_last_assistant_message(retried)

        return body
