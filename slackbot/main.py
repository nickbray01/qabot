import asyncio
import hashlib
import hmac
import os
import time
from collections import defaultdict

from fastapi import FastAPI, Header, HTTPException, Request
from slack_sdk.web.async_client import AsyncWebClient
from dotenv import load_dotenv

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

    # ── Agent stub — replace this with LangGraph call ──────────────────
    response = await run_agent(
        user_text=user_text,
        history=conversation_history[thread_ts],
        thread_ts=thread_ts,
    )
    # ────────────────────────────────────────────────────────────────────────

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


# ── Agent stub ───────────────────────────────────────────────────────────────
# Drop LangGraph agent here later.
# `history` is the full conversation so far — pass it to your agent for multi-turn context.

async def run_agent(user_text: str, history: list, thread_ts: str) -> str:
    # TODO: replace with your actual agent
    return f"Hello! You asked: *{user_text}*\n\n_(agent not yet wired up)_"


# ```

# And your `.env`:
# ```
# SLACK_BOT_TOKEN=xoxb-...
# SLACK_SIGNING_SECRET=...
# ```

# And `.gitignore` — do this on day 1:
# ```
# .env
# data/
# __pycache__/
# *.pyc
# .chroma/