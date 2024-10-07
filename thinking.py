"""
title: Collapsible Thought Filter
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 0.1.0
license: AGPL-3.0+, MIT
required_open_webui_version: 0.3.32
"""

#########################################################
# OpenWebUI Filter that collapses model reasoning/thinking into a
# separate section in the reply.

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
</details>
"""

DETAIL_DELETION_REGEX = r"</?details>[\s\S]*?</details>"

class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        thought_title: str = Field(
            default="Thought Process",
            description="Title for the collapsible reasoning section."
        )
        thought_tag: str = Field(
            default="thinking",
            description="The XML tag for model thinking output."
        )
        output_tag: str = Field(
            default="output",
            description="The XML tag for model final output."
        )
        use_thoughts_as_context: bool = Field(
            default=False,
            description=("Include previous thought processes as context for the AI. "
                         "Disabled by default.")
        )
        pass

    def __init__(self):
        self.valves = self.Valves()

    def _create_thought_regex(self) -> str:
        tag = self.valves.thought_tag
        return f"<{tag}>(.*?)</{tag}>"

    def _create_thought_tag_deletion_regex(self) -> str:
        tag = self.valves.thought_tag
        return "</?{{THINK}}>[\s\S]*?</{{THINK}}>".replace("{{THINK}}", tag)

    def _create_output_tag_deletion_regex(self) -> str:
        tag = self.valves.output_tag
        return r"</?{{OUT}}>[\s\S]*?</{{OUT}}>".replace("{{OUT}}", tag)

    def _enclose_thoughts(self, messages: List[Dict[str, str]]) -> None:
        if not messages:
            return

        # collapsible thinking process section
        thought_regex = self._create_thought_regex()
        reply = messages[-1]["content"]
        thoughts = re.findall(thought_regex, reply, re.DOTALL)
        thoughts = "\n".join(thoughts).strip()
        enclosure = THOUGHT_ENCLOSURE.replace("{{THOUGHT_TITLE}}", self.valves.thought_title)
        enclosure = enclosure.replace("{{THOUGHTS}}", thoughts).strip()

        # remove processed thinking and output tags.
        # some models do not close output tags properly.
        thought_tag_deletion_regex = self._create_thought_tag_deletion_regex()
        output_tag_deletion_regex = self._create_output_tag_deletion_regex()
        reply = re.sub(thought_tag_deletion_regex, "", reply, count=1)
        reply = re.sub(output_tag_deletion_regex, "", reply, count=1)
        reply = reply.replace(f"<{self.valves.output_tag}>", "", 1)
        reply = reply.replace(f"</{self.valves.output_tag}>", "", 1)

        # prevents empty thought process blocks when filter used with
        # malformed LLM output.
        if len(enclosure) > 0:
            reply = f"{enclosure}\n{reply}"

        messages[-1]["content"] = reply

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
