#!/usr/bin/env python3
"""Local stdin/stdout worker for the kev-0.5b decision model (see kev_use.py).

Not a server: it is a child process of the GUI app, talking JSON lines over pipes.
Protocol (one JSON object per line on stdin, one per line on stdout):

    -> {"content": "translated transcript to judge"}
    <- {"label": "productive|unproductive|neutral", "confidence": 0.83,
        "probabilities": {"productive": 0.83, ...}}

Errors are reported as {"error": "..."} instead of crashing the loop. The
checkpoint is loaded once and the worker stays resident until the app kills it
or the stdin stream closes.
"""
import json
import os
import sys

import kev_use  # bootstraps into src/kev-source/.venv automatically

# resolve_run() treats a run name as a directory relative to the CWD, falling back
# to the Hugging Face Hub (which is offline for us) when it is missing — so this
# must be absolute or the checkpoint only loads from one working directory.
RUN_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kev-source", "runs", "kev")
QUESTION_ID = "productivity"

QUESTION = {
    "type": "choice",
    "instructions": "Judge this spoken utterance from someone at work.",
    "criteria": {
        "productive": (
            "focused work talk: solving, planning, checking, explaining "
            "or asking about work"
        ),
        "unproductive": (
            "complaints, venting, slacking off, distraction"
        ),
        "neutral": (
            "small talk, greetings, weather, unrelated chat"
        ),
    },
}


def main():
    kev = kev_use.load(run=RUN_NAME)     # device/dtype/temperature from the checkpoint defaults
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == '"quit"' or line == "quit":
            break
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"bad request: {e}"}), flush=True)
            continue
        content = msg.get("content")
        if not content:
            print(json.dumps({"error": "empty content"}), flush=True)
            continue
        try:
            answer = kev.ask(content, {QUESTION_ID: QUESTION})
            a = answer[QUESTION_ID]
            print(json.dumps({"label": str(a.result),
                              "confidence": a.confidence,
                              "probabilities": a.probabilities}), flush=True)
        except Exception as e:   # keep the loop alive, the caller decides what to do
            print(json.dumps({"error": repr(e)}), flush=True)


if __name__ == "__main__":
    main()
