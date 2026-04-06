# Design Decisions
File manually written by me (Nicholas Bray), no ai used.


# Task:
Create an agent that can answer questions in a Slack channel, using information from a provided database.


# Agent Implementation:
### Model: 
- gpt-4o
### Framework: 
- [LangGraph ReAct](https://docs.langchain.com/oss/python/langchain/agents). Standard ReAct loop - model reasons, calls tools, loops until finished.
- Nodes: 
    - call_model - invokes gpt 4o with system prompt + tools
    - call_tools - executes whichever tools the model asked for

## Tools Provided: 
### think
Call this first for complicated problems (hard examples provided). Allows the agent to decompose abstract / multistep problems into actionable steps and multiple structured requests to the DB

### search
The database provided us with FTS artifacts. Full text search turned out to be a very for matching search queries to documents and deciding what the agent should read for context.

### read
Return an entire document from the DB. Once a match has been found for a keyword, read allows the agentto view the full document. This lets the agent gather all available information and ground its answers on everything the database offers. Since we already have the search function to narrow down the search space, and documents are only ~500-2000 tokens, we can pass everything into the LLM and not worry about context window.

### lookup
Structured row lookup for enumerating customers by attribute. This allows the LLM to see patterns across multiple customers that share an attribute in the DB. E.g. all customers in North America, all customers where crm_stage = "at_risk", etc. 

### sql_query
Helpful for any database exploration not explicitly provided as a tool. Leverage the LLM's decision making to merge tables, view sections of the database in ways I didn't originally intend but it could need to get to an answer. 

NOTE: having an AI Security background, I worry about this tool. I enforced a software-level policy (as opposed to an LLM-level policy) confirming that ONLY single execution queries starting with SELECT are permissible through this tool. This makes the database read-only. Even if you successfully jailbreak the LLM, then to try to execute this tool with `DROP * FROM customers`, the tool's logic wouldn't allow it.

# Security
Since I'm coming from PANW's AI Runtime Security, the first thing I thought of was how risky direct access to a database could be. As mentioned above in the `sql_query` tool's note, I made sure the database is read-only. 

### Tool Leakage
The agent is susceptible to an attack called tool leakage - if you ask nicely, the agent will tell you all of the tools it has access to, as well as their schemas.

### Database Schema
If you leak the agent's tools, you will immediately notice `sql_query` as a promising attack vector. A user-definable SQL query on the prod database? Hello?? Basically a neon sign saying "hackers welcome!". You can tell the agent to run `sql_query` and give you the names of all tables in the database, and it will happily provide that information. 

I later realized you don't even need to know the names of the tools to get this info. Since the DB schema is part of the system prompt, it has this knowledge out of the gate. If this were a true production database and there were ways to connect to it outside of the agent's tools (which I've protected against writes/deletes), exposing this information is equivalent to giving away the keys to the kingdom.

### Fixes
1. Read-only on the DB queries (applied)
2. Guardrails so that jailbreak requests are not passed on to the agent. LlamaGuard, GCP Model Armor, or some equivalent run in-line before the thinking step. (not applied)

### .env best practices
Stored all variables in a .env file which is not tracked, so we're not leaking any keys to GitHub! 


# Eval + Judge Agents & Auto-Improvements

Once I had the tools to explore the database, I needed some way to evaluate whether my answer was correctly answering the questions. It could be the machine learning engineer in me, but I needed to visualize my accuracy, a summary of the changes made between versions, and my progress over time to know whether I was headed in the right direction. I capture the whole improvement journey at <update_website_link_later>

### Ground Truth
I used Claude Code to generate answers to the questions. I had it source the documents that would be needed in order to come up with those answers, and I stored that data in a structured format in `/evals/test_cases.py`. As a backup, I created `evals/eval_eval.py` so that I could manually look at the sources and confirm it wasn't hallucinating. I always manually check a couple samples to make sure my 'all powerful ai model' isn't hallucinating when it generates the ground truth I'm going to trust for my training data.

### Judge
I created a second agent to evaluate whether the answer provided by MY agent matched the ground truth stored in `test_cases`. It looks for:
1. did the agent find the right sources?
2. did the agent's answer contain the expected answer?
3. how many tool calls did it take?
4. reasoning - why the answer is correct, and if not what went wrong along the way

### Reporting and Versioning
With the judge agent, I could understand how well we performed on any test case. With `evals/run_evals.py`, I could generate a report about the performance metrics for ALL test cases. This allowed me to see what percent of test cases I was correctly answering, and what percent I was missing. This seems to me like the only way to know your agent's performance. And it closely resembles ML work I've done elsewhere. 

### Synthetic Test Set Generation to Ensure Generalizability
Solving all 7 provided questions is a great start, and relatively simple once I have reports telling me what was missed. But it's not enough - what if the user wants to know something in a part of the database that wasn't covered by the first 7 questions? I used Claude Code to come up with 12 additional questions, in as far-apart-as-possible parts of the DB. Things that the original 7 covered, that wouldn't let me totally overfit my tools to the provided sample (if you read the blog above you'll see that happened early on.)

### AI Principles
I keep constant logs of every decision made along the way in the `DECISIONS.md` file in this same directory. As I work with Claude Code on the implementation, I regularly have it update this file with the tradeoffs I'm making, discoveries I've seen in the system, and what's motivating architectural decisions. 

I designed this eval/judge framework to make it easy to feed Claude Code shortcomings on the test set, and enable it to efficiently make necessary improvements in the agent's tools, rules, and metaprompts. I applied ML best-practices for making my system as performant as possible without overfitting to the test set, but I did it from a higher leverage POV with my evaluator agent and my coding/retraining agent. This is how I was able to turn around 8 model versions in 3 days outside of my full time job. I was the architect, the agents helped me implement and improve.


# Slack Implementation:
- Authenticated with HMAC SHA-256 signature verification (timing-safe)
- Responds with a "🤔 thinking..." message immediately since Slack requires a `200 response` within 3 seconds. Then hand off the agent processing to `asyncio.create_task()` to process the longer-running agent information.
- Provide information any time a tool is called for intermitent updates
- Update the original message with the agents so we don't overwhelm the thread with tool-calling messages. Provide a summary at the end of all the tools used.
- Use `thread_ts` so that multi-turn conversations can be handled within a thread and pass message history back.