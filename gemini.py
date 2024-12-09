"""
title: Gemini Protocol Tool
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 0.1.2
license: AGPL-3.0+
required_open_webui_version: 0.4.3
requirements: ignition-gemini
"""
import re
import ignition
from ignition import RedirectResponse, SuccessResponse
from pydantic import BaseModel, Field
from typing import Optional

def result_instructions(url: str, redirect: bool=False) -> str:
    content_instructions = (
        "Report the content to the user and answer their question."
        "Do not make assumptions. This is the content you need. "
        "Even if you think it's not the right content, report the "
        "content anyway. Do not say you think it's the wrong content."
        "You MUST use this content."
    )

    return ("# Gemini Content Fetch Result\n"
            f"Content was successfully fetched for the URL: {url}\n"
            ) + content_instructions + "\n\n"

def instructions(url: str, redirect: bool=False) -> str:
    return result_instructions(url, redirect) + (
        "Here are more instructions you must follow. "
        "Render all Gemini links as Markdown links. Examples:\n\n"
        " - [gemini://example.com](gemini://example.com)\n"
        " - [a gemini capsule](gemini://example.com)\n"
        " - [my personal capsule](gemini://example.com/personal)\n\n"
        "A Gemini link starts with => on the line, and then the URL follows:\n\n"
        " - `=> /page A Gemini Page`: A relative link to `/page` titled `A Gemini Page`\n"
        " - `=> gemini://example.com/place` An absolute link with no URL title \n\n"
        "When rendering relative links, always make them absolute links.\n"
        "If the link has a title, render the link title verbatim:\n"
        " - `=> gemini://example.com My Page` becomes `[My Page](gemini://example.com)`\n"
        " - `=> gemini://example.com` becomes `[gemini://example.com](gemini://example.com)`.\n"
        " - `=> gemini://example.com 🐂 My Page` becomes `[🐂 My Page](gemini://example.com)`\n\n"
        "The Gemtext content is below in the code block."
        )

def correct_url(url: str) -> str:
    if url.startswith("gemini://http://"):
        match = re.match(r'gemini://http://(.+)', url)
        if match:
            return f"gemini://{match.group(1)}"
        return url

    if url.startswith("gemini://https://"):
        match = re.match(r'gemini://https://(.+)', url)
        if match:
            return f"gemini://{match.group(1)}"
        return url

    if url.startswith("https://"):
        match = re.match(r'https://(.+)', url)
        if match:
            return f"gemini://{match.group(1)}"
        return url

    if url.startswith("http://"):
        match = re.match(r'http://(.+)', url)
        if match:
            return f"gemini://{match.group(1)}"
        return url

    if not url.startswith("gemini://"):
        return f"gemini://{url}"

    return url


def fetch(gemini_url: str, correct_urls: bool=False, prev_url: Optional[str]=None, redirects: int=0) -> dict:
    if redirects > 5:
        return {
            "success": False,
            "content": f"Too many redirects (ended at {gemini_url})",
            "redirected": prev_url is not None
        }

    if correct_urls and not prev_url:
        corrected_url = correct_url(gemini_url)
        if corrected_url != gemini_url:
            print(f"[Gemini] URL '{gemini_url}' corrected to '{corrected_url}'")
            gemini_url = corrected_url

    if not prev_url:
        print(f"[Gemini] Fetching: {gemini_url}")
    else:
        print(f"[Gemini] Fetching: {gemini_url} (redirected from {prev_url})")

    try:
        response = ignition.request(gemini_url, raise_errors=True, referer=prev_url)

        if isinstance(response, SuccessResponse):
            return {
                "success": True,
                "content": handle_content(response),
                "redirected": prev_url is not None
            }
        elif isinstance(response, RedirectResponse):
            redirect_url = response.data()
            return fetch(redirect_url, correct_urls, gemini_url, redirects + 1)
        else:
            print(f"[Gemini] Unhandled {response.status} code for '{gemini_url}'")
            message = (f"Tell the user there was a {response.status} status code. "
                       f"Support for handling {response.status} is not implemented yet.")
            return { "success": False, "content": message, "redirected": prev_url is not None }
    except Exception as e:
        print(f"[Gemini] error: {e}")
        message = f"Tell the user there was an error fetching the page: {e}"
        return {
            "success": False,
            "content": message,
            "redirected": prev_url is not None
        }

def handle_content(resp: SuccessResponse) -> str:
    try:
        mime_type, encoding = resp.meta.split(";")
    except ValueError:
        mime_type = resp.meta

    mime_type = mime_type.strip()

    if mime_type == "text/gemini":
        return resp.data().strip()
    else:
        raise ValueError(f"Not yet able to handle MIME type {mime_type}")

class Tools:
    class Valves(BaseModel):
        attempt_url_correction: str = Field(
            default=True, description="Attempt to correct malformed URLs (default enabled)."
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = None

    def get_gemini_page(self, gemini_url: str, __event_emitter__) -> str:
        """
        Fetches Gemini capsules, content, and web pages over Gemini Protocol.
        Use this if the user requests a gemini:// URL.
        :param gemini_url: The URL to fetch. The URL MUST begin with gemini://.
        :return: The fetched data as Markdown.
        """
        resp = fetch(gemini_url, correct_urls=self.valves.attempt_url_correction)
        if resp["success"]:
            result_instructions = instructions(gemini_url, redirect=resp["redirected"])
            return f"{result_instructions}\n\n```\n{resp['content']}\n```"
        else:
            return resp["content"]
