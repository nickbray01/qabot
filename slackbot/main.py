import asyncio
import hashlib
import hmac
import os
import re
import time
from collections import defaultdict

from fastapi import FastAPI, Header, HTTPException, Request
from slack_sdk.web.async_client import AsyncWebClient
from dotenv import load_dotenv
from search_agent import run_agent

load_dotenv()

app = FastAPI()
slack = AsyncWebClient(token=os.environ["SLACK_BOT_TOKEN"])

# In-memory thread history — swap for Redis/DB later
# { thread_ts: [{"role": "user"|"assistant", "content": str}, ...] }
conversation_history: dict[str, list] = defaultdict(list)


# ── Signature verification ──────────────────────────────────────────────────

def verify_slack_signature(body: bytes, timestamp: str, signature: str) -> bool:
    if abs(time.time() - float(timestamp)) > 300:
        return False
    sig_basestring = f"v0:{timestamp}:{body.decode()}".encode()
    expected = "v0=" + hmac.new(
        os.environ["SLACK_SIGNING_SECRET"].encode(),
        sig_basestring,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


# Routes

@app.post("/slack/events")
async def slack_events(
    request: Request,
    x_slack_request_timestamp: str = Header(...),
    x_slack_signature: str = Header(...),
):
    body = await request.body()

    if not verify_slack_signature(body, x_slack_request_timestamp, x_slack_signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()

    # One-time URL verification handshake from Slack
    if payload.get("type") == "url_verification":
        return {"challenge": payload["challenge"]}

    event = payload.get("event", {})

    # Only handle @mentions, skip bot's own messages
    if event.get("type") == "app_mention" and not event.get("bot_id"):
        asyncio.create_task(handle_mention(event))

    return {"ok": True}  # must ack within 3s — agent runs in background


# Core handler

async def handle_mention(event: dict):
    channel = event["channel"]
    user = event["user"]
    raw_text = event.get("text", "")
    event_ts = event["ts"]

    # Use thread_ts if already in a thread, else start one from this message
    thread_ts = event.get("thread_ts") or event_ts

    # Strip the @qabot mention from the text
    user_text = raw_text.split(">", 1)[-1].strip()

    # Append to conversation history for this thread
    conversation_history[thread_ts].append({
        "role": "user",
        "content": user_text,
    })

    # Show a "thinking" message immediately; we'll edit it in-place as tools run
    status_msg = await slack.chat_postMessage(
        channel=channel,
        thread_ts=thread_ts,
        text=":thinking_face: _thinking..._",
    )
    status_ts = status_msg["ts"]

    tool_call_log: list[dict] = []

    async def on_tool_call(tool_name: str, args: dict) -> None:
        tool_call_log.append({"name": tool_name, "args": args})
        label = _TOOL_LABELS.get(tool_name, f"Calling {tool_name}")
        hint = args.get("query") or args.get("name") or args.get("entity") or ""
        text = f":mag: _{label}: `{hint}`_" if hint else f":mag: _{label}..._"
        await slack.chat_update(channel=channel, ts=status_ts, text=text)

    response = await run_agent(
        user_text=user_text,
        history=conversation_history[thread_ts],
        thread_ts=thread_ts,
        on_tool_call=on_tool_call,
    )

    # Edit the status message one final time with a compact tool-use summary
    await slack.chat_update(
        channel=channel,
        ts=status_ts,
        text=_build_tool_summary(tool_call_log),
    )

    # Append agent response to history
    conversation_history[thread_ts].append({
        "role": "assistant",
        "content": response,
    })

    # Trim history if it gets long (keep last 20 turns)
    if len(conversation_history[thread_ts]) > 20:
        conversation_history[thread_ts] = conversation_history[thread_ts][-20:]

    await slack.chat_postMessage(
        channel=channel,
        thread_ts=thread_ts,
        text=_to_mrkdwn(response),
    )


def _to_mrkdwn(text: str) -> str:
    """Convert standard markdown to Slack mrkdwn."""
    # **bold** → *bold*
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text, flags=re.DOTALL)
    # # Heading / ## Heading → *Heading*
    text = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
    # "- item" at line start → "• item"
    text = re.sub(r'^\s*-\s+', '• ', text, flags=re.MULTILINE)
    return text


def _build_tool_summary(tool_call_log: list[dict]) -> str:
    searches: list[str] = []
    lookups: list[str] = []
    read_ids: list[str] = []

    for tc in tool_call_log:
        name = tc["name"]
        args = tc["args"]
        if name == "search":
            q = args.get("query", "")
            if q:
                searches.append(q)
        elif name == "lookup":
            parts = [args.get("entity", "")]
            if args.get("name"):
                parts.append(args["name"])
            lookups.append(" ".join(p for p in parts if p))
        elif name == "read":
            ids = args.get("artifact_ids", [])
            if isinstance(ids, list):
                read_ids.extend(ids)
            elif ids:
                read_ids.append(str(ids))
        # think and sql_query are intentionally omitted from the summary

    parts: list[str] = []
    if searches:
        quoted = ", ".join(f'"{q}"' for q in searches)
        parts.append(f"searched {quoted}")
    if lookups:
        parts.append(f"looked up {', '.join(lookups)}")
    if read_ids:
        id_list = ", ".join(f"`{i}`" for i in read_ids)
        parts.append(f"read {len(read_ids)} doc(s): {id_list}")

    if not parts:
        return ":mag: _Done._"
    return ":mag: _" + " · ".join(parts) + "_"


_TOOL_LABELS: dict[str, str] = {
    "search_artifacts": "Searching artifacts",
    "find_pattern_across_customers": "Scanning across all customers",
    "customer_artifacts": "Looking up customer",
    "scenario_summary_tool": "Loading scenario",
    "artifact_full_text": "Reading document",
    "sql_query": "Running query",
}

