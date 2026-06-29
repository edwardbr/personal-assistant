#!/usr/bin/env python3
"""Small stdio MCP server for voice-assistant utility tools."""
from __future__ import annotations

import datetime as dt
import email.utils
import html
import json
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests


PROTOCOL_VERSION = "2025-06-18"


def write(msg: dict[str, Any]):
    sys.stdout.write(json.dumps(msg, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def result(msg_id: Any, value: dict[str, Any]):
    write({"jsonrpc": "2.0", "id": msg_id, "result": value})


def error(msg_id: Any, code: int, message: str):
    write({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}})


def text_result(text: str, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def get_time(args: dict[str, Any]) -> str:
    timezone = str(args.get("timezone") or "").strip()
    if timezone and timezone.lower() not in {"local", "system"}:
        try:
            now = dt.datetime.now(ZoneInfo(timezone))
            zone_label = timezone
        except ZoneInfoNotFoundError:
            now = dt.datetime.now().astimezone()
            zone_label = f"local time (unknown timezone {timezone!r})"
    else:
        now = dt.datetime.now().astimezone()
        zone_label = now.tzname() or "local time"
    return f"{now:%A, %d %B %Y, %H:%M:%S} {zone_label}"


def _flatten_related(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if "Topics" in item:
            out.extend(_flatten_related(item.get("Topics") or []))
        else:
            out.append(item)
    return out


def _duckduckgo(query: str, max_results: int) -> list[str]:
    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "no_redirect": "1",
        "skip_disambig": "1",
    }
    response = requests.get("https://api.duckduckgo.com/", params=params, timeout=8)
    response.raise_for_status()
    data = response.json()
    results: list[str] = []
    if data.get("Answer"):
        results.append(str(data["Answer"]))
    if data.get("AbstractText"):
        text = str(data["AbstractText"])
        if data.get("AbstractURL"):
            text += f" ({data['AbstractURL']})"
        results.append(text)
    if data.get("Definition"):
        text = str(data["Definition"])
        if data.get("DefinitionURL"):
            text += f" ({data['DefinitionURL']})"
        results.append(text)
    for item in _flatten_related(data.get("RelatedTopics") or []):
        if len(results) >= max_results:
            break
        text = str(item.get("Text") or "").strip()
        if text:
            if item.get("FirstURL"):
                text += f" ({item['FirstURL']})"
            results.append(text)
    return results[:max_results]


def _wikipedia(query: str, max_results: int) -> list[str]:
    params = {
        "action": "opensearch",
        "search": query,
        "limit": max_results,
        "namespace": "0",
        "format": "json",
    }
    response = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=8)
    response.raise_for_status()
    data = response.json()
    titles = data[1] if len(data) > 1 else []
    descriptions = data[2] if len(data) > 2 else []
    urls = data[3] if len(data) > 3 else []
    results: list[str] = []
    for i, title in enumerate(titles[:max_results]):
        desc = descriptions[i] if i < len(descriptions) and descriptions[i] else ""
        url = urls[i] if i < len(urls) else ""
        text = str(title)
        if desc:
            text += f": {desc}"
        if url:
            text += f" ({url})"
        results.append(text)
    return results


def _strip_markup(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


RSS_FEEDS = {
    "bbc": "https://feeds.bbci.co.uk/news/rss.xml",
    "bbc_uk": "https://feeds.bbci.co.uk/news/uk/rss.xml",
    "bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "bbc_business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "bbc_technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "bbc_science": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
}


def _pick_news_feed(args: dict[str, Any]) -> tuple[str, str]:
    source = str(args.get("source") or "").strip().lower()
    topic = str(args.get("topic") or "").strip().lower()
    country = str(args.get("country") or "").strip().lower()
    text = " ".join([source, topic, country])

    if source in RSS_FEEDS:
        key = source
    elif any(word in text for word in ("uk", "united kingdom", "britain", "british", "england", "scotland", "wales")):
        key = "bbc_uk"
    elif any(word in text for word in ("world", "international", "global")):
        key = "bbc_world"
    elif "business" in text or "finance" in text:
        key = "bbc_business"
    elif "tech" in text or "technology" in text:
        key = "bbc_technology"
    elif "science" in text or "environment" in text:
        key = "bbc_science"
    else:
        key = "bbc"
    return key, RSS_FEEDS[key]


def news_headlines(args: dict[str, Any]) -> str:
    try:
        max_results = max(1, min(10, int(args.get("max_results", 5))))
    except (TypeError, ValueError):
        max_results = 5
    include_links = bool(args.get("include_links", False))

    feed_key, feed_url = _pick_news_feed(args)
    response = requests.get(feed_url, timeout=8)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    channel = root.find("channel")
    if channel is None:
        return f"No RSS channel found at {feed_url}"

    source_title = channel.findtext("title") or feed_key
    pub_date = channel.findtext("lastBuildDate") or channel.findtext("pubDate") or ""
    if pub_date:
        try:
            pub_dt = email.utils.parsedate_to_datetime(pub_date).astimezone()
            pub_date = f" Updated {pub_dt:%H:%M %Z, %d %B %Y}."
        except Exception:
            pub_date = f" Updated {pub_date}."

    lines = [f"Top {min(max_results, 10)} headlines from {source_title}.{pub_date}"]
    count = 0
    for item in channel.findall("item"):
        title = _strip_markup(item.findtext("title") or "")
        if not title:
            continue
        description = _strip_markup(item.findtext("description") or "")
        link = (item.findtext("link") or "").strip()
        count += 1
        line = f"{count}. {title}"
        if description and description != title:
            line += f" - {description}"
        if include_links and link:
            line += f" ({link})"
        lines.append(line)
        if count >= max_results:
            break

    if count == 0:
        return f"No headlines found in {source_title} ({feed_url})."
    return "\n".join(lines)


def web_search(args: dict[str, Any]) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "No search query was provided."
    try:
        max_results = max(1, min(5, int(args.get("max_results", 3))))
    except (TypeError, ValueError):
        max_results = 3

    results: list[str] = []
    try:
        results = _duckduckgo(query, max_results)
    except Exception as e:
        results.append(f"DuckDuckGo lookup failed: {e}")

    if not results or all(item.startswith("DuckDuckGo lookup failed:") for item in results):
        try:
            results.extend(_wikipedia(query, max_results))
        except Exception as e:
            results.append(f"Wikipedia lookup failed: {e}")

    if not results:
        encoded = urllib.parse.quote_plus(query)
        return f"No concise result found. Search URL: https://duckduckgo.com/?q={encoded}"

    lines = [f"Web results for {query!r}:"]
    lines.extend(f"{i}. {item}" for i, item in enumerate(results[:max_results], 1))
    return "\n".join(lines)


TOOLS = {
    "get_time": {
        "description": "Get the current date and time. Optional timezone is an IANA name such as Europe/London.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "IANA timezone name, or local/system."}
            },
        },
        "handler": get_time,
    },
    "web_search": {
        "description": "Look up concise current information from the internet.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
            },
            "required": ["query"],
        },
        "handler": web_search,
    },
    "news_headlines": {
        "description": "Get current news headlines from RSS feeds. Use this for news, headlines, top stories, or breaking news.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "Optional country/region, e.g. uk, world, international.",
                },
                "topic": {
                    "type": "string",
                    "description": "Optional topic: business, technology, science, world.",
                },
                "source": {
                    "type": "string",
                    "description": "Optional feed key: bbc, bbc_uk, bbc_world, bbc_business, bbc_technology, bbc_science.",
                },
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                "include_links": {"type": "boolean", "default": False},
            },
        },
        "handler": news_headlines,
    },
}


def handle(msg: dict[str, Any]):
    msg_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}

    if method == "initialize":
        result(
            msg_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "whisper-bridge-local-tools", "version": "0.1"},
            },
        )
    elif method == "notifications/initialized":
        return
    elif method == "tools/list":
        result(
            msg_id,
            {
                "tools": [
                    {
                        "name": name,
                        "description": spec["description"],
                        "inputSchema": spec["inputSchema"],
                    }
                    for name, spec in TOOLS.items()
                ]
            },
        )
    elif method == "tools/call":
        name = str(params.get("name") or "")
        args = params.get("arguments") or {}
        if name not in TOOLS:
            result(msg_id, text_result(f"Unknown tool: {name}", is_error=True))
            return
        try:
            text = TOOLS[name]["handler"](args)
            result(msg_id, text_result(text))
        except Exception as e:
            result(msg_id, text_result(f"{name} failed: {e}", is_error=True))
    else:
        error(msg_id, -32601, f"Unknown method: {method}")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            handle(json.loads(line))
        except Exception as e:
            sys.stderr.write(f"local MCP server error: {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
