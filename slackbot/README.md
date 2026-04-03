# slackbot

FastAPI webhook server that receives Slack `app_mention` events and routes them to the LangGraph agent. This module is complete and stable — it does not need changes once the agent is wired in.

## How it works

1. Slack POSTs to `/slack/events` when a user mentions `@qabot`
2. The server validates the Slack signature (HMAC-SHA256, timing-safe) and checks the timestamp is within 5 minutes
3. HTTP 200 is returned immediately to satisfy Slack's 3-second deadline
4. `handle_mention` runs as an async background task
5. A `:thinking_face: thinking...` message is posted to the thread right away
6. `run_agent()` is called with the user's text and full thread history
7. The agent's response is posted back to the same thread

Conversation history is kept in memory, keyed by `thread_ts`, capped at 20 turns. Swap for Redis or a SQLite-backed store when persistence is needed.

## Prerequisites

- Python 3.11+
- A Slack app with `app_mentions:read` and `chat:write` bot token scopes
- ngrok (or equivalent) to expose localhost during development

## Setup

### 1. Install dependencies

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install fastapi uvicorn slack-sdk python-dotenv
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set:

```
SLACK_BOT_TOKEN=xoxb-your-token-here
SLACK_SIGNING_SECRET=your-signing-secret-here
```

The `.env` file must live at the **project root** (same level as `slackbot/`), not inside this folder. `load_dotenv()` resolves paths relative to the working directory.

### 3. Create your Slack app

Go to [api.slack.com/apps](https://api.slack.com/apps) and create a new app from scratch.

| Step | Where | Action |
|------|-------|--------|
| Signing secret | Basic Information → App Credentials | Copy the Signing Secret into `SLACK_SIGNING_SECRET` |
| Bot scopes | OAuth & Permissions → Bot Token Scopes | Add `app_mentions:read` and `chat:write` |
| Install | OAuth & Permissions | Install to Workspace, copy the `xoxb-...` token into `SLACK_BOT_TOKEN` |
| App Home | App Home | Toggle "Always Show My Bot as Online", set display name to `qabot` |
| Events | Event Subscriptions | Enable — leave the URL blank until ngrok is running (step 4) |
| Bot events | Event Subscriptions → Subscribe to Bot Events | Add `app_mention` and save |

### 4. Start the server

```bash
uvicorn slackbot.main:app --reload --port 8000
```

You should see `Application startup complete.`

### 5. Start the ngrok tunnel

In a second terminal:

```bash
ngrok http 8000
```

Copy the `Forwarding` URL (e.g. `https://abc-123.ngrok-free.app`).

### 6. Connect Slack to your server

1. Go back to your Slack app → Event Subscriptions
2. Paste your ngrok URL + `/slack/events` into the Request URL field:
   ```
   https://abc-123.ngrok-free.app/slack/events
   ```
3. Slack sends a challenge — the server handles it automatically. Wait for the green **Verified** checkmark.
4. Save Changes.

### 7. Invite the bot and test

In any Slack channel:

```
/invite @qabot
@qabot hello
```

You should see a "thinking..." message followed by the stub response in a thread.

## Wiring in the agent

Replace the stub in `run_agent()` ([main.py:120](main.py#L120)) with your LangGraph call. The function signature stays the same:

```python
async def run_agent(user_text: str, history: list, thread_ts: str) -> str:
    ...
```

The Slack layer does not need to change.

## Known gotchas

| Issue | Fix |
|-------|-----|
| ngrok URL resets on restart | Re-paste the new URL in Event Subscriptions. Consider reserving a free static domain in the ngrok dashboard. |
| Duplicate responses from Slack retries | Already handled — `asyncio.create_task()` ensures HTTP 200 is returned before any slow work begins. |
| Bot not responding in a channel | Run `/invite @qabot` in that channel. Installing to the workspace is not enough. |
| OAuth scopes not taking effect | Adding or changing scopes requires reinstalling the app (OAuth & Permissions → Reinstall to Workspace). |
