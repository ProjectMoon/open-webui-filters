"""
title: Output Sanitization Filter
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 0.1.2
license: AGPL-3.0+
required_open_webui_version: 0.3.9
"""

# Documentation: https://git.agnos.is/projectmoon/open-webui-filters

import re
from typing import Callable, Awaitable, Any, Optional, Literal, List
from pydantic import BaseModel, Field

def strip_prefixes(message: str, prefixes: List[str]):
    for prefix in prefixes:
        if message.startswith(prefix):
            message = message.replace(prefix, "", 1)
    return message

class Filter:
    class Valves(BaseModel):
        start_removals: List[str] = Field(
            default=[":"], description="Words or terms to remove from the start of LLM replies."
        )

        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    async def replace_message(self, message):
        await self.event_emitter({
            "type": "replace",
            "data": {
                "content": message
            }
        })


    async def outlet(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
    ) -> dict:
        """Remove words from the start and end of LLM replies."""
        self.event_emitter = __event_emitter__

        if len(body["messages"]) == 0:
            return body

        last_reply: dict = body["messages"][-1]
        last_reply = last_reply["content"].strip()
        replaced_message = strip_prefixes(last_reply, self.valves.start_removals)

        if replaced_message != last_reply:
            body["messages"][-1]["content"] = replaced_message
            await self.replace_message(replaced_message)

        return body
