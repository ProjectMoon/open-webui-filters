"""
title: Gemini Protocol Tool
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 0.0.1
license: AGPL-3.0+
required_open_webui_version: 0.4.3
requirements: ignition-gemini
"""
import re
import ignition
from pydantic import BaseModel, Field

def instructions() -> str:
    return ("Render all Gemini links as Markdown links. Examples:\n\n"
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
        try:
            print(f"[Gemini] Fetching: {gemini_url}")
            if self.valves.attempt_url_correction:
                corrected_url = correct_url(gemini_url)
                if corrected_url != gemini_url:
                    print(f"[Gemini] URL '{gemini_url}' corrected to '{corrected_url}'")
                    gemini_url = corrected_url

            response = ignition.request(gemini_url, raise_errors=True)

            if str(response.status).startswith("2"):
                content = response.data()
                return f"{instructions()}\n\n ```\n{content.strip()}\n```"
            else:
                print(f"[Gemini] Unhandled {response.status} code for '{gemini_url}'")
                return (f"Tell the user there was a {response.status} status code. "
                        f"Support for handling {response.status} is not implemented yet.")
        except Exception as e:
            print(f"[Gemini] error: {e}")
            return f"Tell the user there was an error fetching the page: {e}"
