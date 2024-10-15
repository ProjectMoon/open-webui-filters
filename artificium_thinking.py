"""
title: Artificium Thought Filter
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 0.1.1
license: AGPL-3.0+, MIT
required_open_webui_version: 0.3.32
"""

#########################################################
# OpenWebUI Filter that collapses model reasoning/thinking into a
# separate section in the reply. This is specificially for the
# Artificium model, based on Llama 3.1. It outputs its thought
# processes broken by markdown horizontal rules. Usually, it outputs a
# basic thought process followed by a breakdown. GENERALLY, text below
# the last horizontal line is the final answer.
#
# Based on the Add or Delete Text Filter by anfi.
# https://openwebui.com/f/anfi/add_or_delete_text
#
# Therefore, portions of this code are licensed under the MIT license.
# The modifications made for "thought enclosure" etc are licensed
# under the AGPL using the MIT's sublicensing clause.
#
# For those portions under the MIT license, the following applies:
#
# MIT License
#
# Copyright (c) 2024 anfi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#########################################################

from typing import Optional, Dict, List
import re
from pydantic import BaseModel, Field

THOUGHT_ENCLOSURE = """
<details>
<summary>{{THOUGHT_TITLE}}</summary>
{{THOUGHTS}}

---

</details>
"""

DETAIL_DELETION_REGEX = r"</?details>[\s\S]*?</details>"

class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        task_title: str = Field(
            default="Task Discovery",
            description="Title for the collapsible task reasoning section."
        )
        breakdown_title: str = Field(
            default="Thought Process",
            description="Title for the collapsible reasoning breakdown section."
        )
        use_thoughts_as_context: bool = Field(
            default=False,
            description=("Include previous thought processes as context for the AI. "
                         "Disabled by default.")
        )
        pass

    def __init__(self):
        self.valves = self.Valves()

    def _parse_reply(self, messages: List[Dict[str, str]]) -> dict:
        reply = messages[-1]["content"]
        sections = [section.strip() for section in reply.split('\n\n---\n\n')]
        sections = [section for section in sections if section]
        print(f"[Artificium Filter] Parsed {len(sections)} section(s)")

        # a few different situations.
        # 1. 3+ sections = initial thoughts, breakdown, final output.
        # 2. 2 sections = thoughts, final output
        # 3. 1 section or 0 sections = do nothing
        if len(sections) >= 3:
            return {
                "initial": sections[0],
                "breakdown": "\n\n---\n\n".join(sections[1:-1]),
                "final": sections[-1]
            }
        elif len(sections) == 2:
            return {
                "initial": sections[0],
                "breakdown": None,
                "final": sections[1]
            }
        else:
            return {
                "initial": None,
                "breakdown": None,
                "final": reply
            }

    def _enclose_thoughts(self, messages: List[Dict[str, str]]) -> None:
        if not messages:
            return

        parsed_reply = self._parse_reply(messages)
        final_reply = ""

        if parsed_reply["initial"] is not None:
            initial_thoughts = (THOUGHT_ENCLOSURE
                                .replace("{{THOUGHT_TITLE}}", self.valves.task_title)
                                .replace("{{THOUGHTS}}", parsed_reply["initial"]))
            final_reply = initial_thoughts

        if parsed_reply["breakdown"] is not None:
            breakdown_thoughts = (THOUGHT_ENCLOSURE
                                .replace("{{THOUGHT_TITLE}}", self.valves.breakdown_title)
                                .replace("{{THOUGHTS}}", parsed_reply["breakdown"]))
            final_reply = f"{final_reply}\n{breakdown_thoughts}"

        if parsed_reply["final"] is not None:
            output = parsed_reply["final"]
            final_reply = f"{final_reply}\n{output}"

        final_reply = final_reply.strip()
        if final_reply:
            messages[-1]["content"] = final_reply

    def _handle_include_thoughts(self, messages: List[Dict[str, str]]) -> None:
        """Remove <details> tags from input, if configured to do so."""
        # <details> tags are created by the outlet filter for display
        # in OWUI.
        if self.valves.use_thoughts_as_context:
            return

        for message in messages:
            message["content"] = re.sub(
                DETAIL_DELETION_REGEX, "", message["content"], count=1
            )

    def inlet(self, body: Dict[str, any], __user__: Optional[Dict[str, any]] = None) -> Dict[str, any]:
        try:
            original_messages: List[Dict[str, str]] = body.get("messages", [])
            self._handle_include_thoughts(original_messages)
            body["messages"] = original_messages
            return body
        except Exception as e:
            print(e)
            return body

    def outlet(self, body: Dict[str, any], __user__: Optional[Dict[str, any]] = None) -> Dict[str, any]:
        try:
            original_messages: List[Dict[str, str]] = body.get("messages", [])
            self._enclose_thoughts(original_messages)
            body["messages"] = original_messages
            return body
        except Exception as e:
            print(e)
            return body
