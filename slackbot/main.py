import asyncio
import hashlib
import hmac
import os
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

    # Show a "thinking" message immediately
    await slack.chat_postMessage(
        channel=channel,
        thread_ts=thread_ts,
        text=":thinking_face: thinking...",
    )

    async def on_tool_call(tool_name: str, args: dict) -> None:
        label = _TOOL_LABELS.get(tool_name, f"Calling {tool_name}")
        hint = args.get("query") or args.get("name_or_id") or args.get("artifact_id") or ""
        text = f":mag: _{label}: `{hint}`_" if hint else f":mag: _{label}..._"
        await slack.chat_postMessage(channel=channel, thread_ts=thread_ts, text=text)

    response = await run_agent(
        user_text=user_text,
        history=conversation_history[thread_ts],
        thread_ts=thread_ts,
        on_tool_call=on_tool_call,
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
        text=response,
    )


_TOOL_LABELS: dict[str, str] = {
    "search_artifacts": "Searching artifacts",
    "find_pattern_across_customers": "Scanning across all customers",
    "customer_artifacts": "Looking up customer",
    "scenario_summary_tool": "Loading scenario",
    "artifact_full_text": "Reading document",
    "sql_query": "Running query",
}

