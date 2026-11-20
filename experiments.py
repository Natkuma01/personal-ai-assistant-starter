"""Prompt experiment: compare two system prompts on the same inputs.

Run it from the terminal:
    python experiment.py

How to use it:
    1. Paste your current system prompt into SYSTEM_PROMPT_A.
    2. Paste it again into SYSTEM_PROMPT_B, then change ONE thing.
    3. Run the file and read the replies.
    4. Write down what happened in experiments.md.
"""

import os
import time

import openai
from dotenv import load_dotenv

MODEL = "openai/gpt-oss-20b"
MAX_REPLY_TOKENS = 300  # same limit as your app
RUNS = 3  # how many times to ask each question, because replies vary

# ---------- Part 1: the two system prompts ----------
# A is what you have now. B is A with ONE change.
SYSTEM_PROMPT_A = """You are expert to train any pokemon. You add a pokemon word in all of your conversation.
Keep your answers short: two or three sentences."""

SYSTEM_PROMPT_B = """You are expert to train any pokemon. You add a pokemon word in all of your conversation. and you add "hai" in the last of every sentence.
Keep your answers short: two or three sentences.
Always end with a follow-up question. Do not say more than 50 words"""

# ---------- Part 2: the test inputs ----------
# The messages a visitor might send. Both prompts get the same ones.
TEST_INPUTS = [
    "What's the best way to learn to code?",
    "When I have Snorelax and Pika stay with me.",
]

# Read your key from the .env file, the same way your app does.
load_dotenv()
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


def get_reply(system_prompt, user_text):
    """Send one message to the model and return its reply."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            max_tokens=MAX_REPLY_TOKENS,
            reasoning_effort="low",  # gpt-oss "thinks" first; keep it short so the reply fits
        )
    except openai.APIError as error:
        return f"(No reply. The model returned an error: {error})"
    time.sleep(1)  # a short pause so we don't send requests too fast
    return response.choices[0].message.content or "(The model sent back an empty reply.)"


# ---------- Part 3: run the experiment ----------
for user_text in TEST_INPUTS:
    print("=" * 70)
    print(f"INPUT: {user_text}")
    print("=" * 70)

    for name, system_prompt in [("SYSTEM PROMPT A", SYSTEM_PROMPT_A), ("SYSTEM PROMPT B", SYSTEM_PROMPT_B)]:
        print(f"\n--- {name} ---")
        for run in range(1, RUNS + 1):
            reply = get_reply(system_prompt, user_text)
            print(f"\n[{name} · run {run} of {RUNS}]")
            print(reply.strip())

    print()

print("Done. Now write down what happened in experiments.md.")