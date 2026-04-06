# slackbot agent

## Links

- **Demo video**: [https://youtu.be/2JFax3_9N0Q](https://youtu.be/2JFax3_9N0Q)
- **Build log**: [https://nickbray-langchain.netlify.app](https://nickbray-langchain.netlify.app) *(password: nickbray)*



FastAPI webhook server that receives Slack `app_mention` events and routes them to the LangGraph agent. This module is complete and stable — it does not need changes once the agent is wired in.

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

Edit `.env` and set:

```
SLACK_BOT_TOKEN=xoxb-your-token-here
SLACK_SIGNING_SECRET=your-signing-secret-here
OPENAI_API_KEY=your-openai-api-key-here
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

# agent evals

The eval harness runs the search agent against a set of ground-truth test cases and grades each answer with an LLM judge.

## Run a single test case directly

Use the runner to invoke the agent and inspect its trace without the full harness:

```python
import asyncio
from evals.runner import run_agent_traced

trace = asyncio.run(run_agent_traced(
    "which customer's issue started after the 2026-02-20 taxonomy rollout, "
    "and what proof plan did we propose to get them comfortable with renewal?"
))

print(trace.answer)
for tc in trace.tool_calls:
    print(tc.name, tc.args)
```

Or run the runner module directly for a quick smoke-test:

```bash
python -m evals.runner
```

## Run the full eval suite

From the project root:

```bash
# All test cases
python evals/run_evals.py

# Specific case IDs
python evals/run_evals.py --id q1_blueharbor_taxonomy q2_verdant_bay_patch

# Filter by difficulty
python evals/run_evals.py --difficulty easy
python evals/run_evals.py --difficulty hard

# Save full JSON results for downstream analysis
python evals/run_evals.py --output report.json
```

The console report includes per-case correctness, answer score, tool call count, and source retrieval. The optional JSON file adds full traces and judge verdicts.
