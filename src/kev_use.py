#!/usr/bin/env python3
"""Programmatic sandbox for the local kev-0.5b decision model.

Loads the checkpoint straight from disk (no server, no network) and answers typed questions
about a piece of content. Questions use the TypeSafe JSON shape:

    noul    {"type": "noul", "instructions": "...", "criteria": {...}}
    choice  {"type": "choice", "instructions": "...", "criteria": {"name": "desc", ...}}
    score   {"type": "score", "instructions": "...", "criteria": ["low", "mid", "high"]}

Example:

    from src.kev import load

    kev = load()  # or load(device="cpu", dtype="fp32", temperature=1.47)

    answer = kev.ask(
        content="The package arrived late and one item was broken.",
        questions={
            "sentiment": {"type": "choice", "criteria": {"negative": "unhappy", "positive": "happy"}},
            "refund":    {"type": "noul", "instructions": "Should we issue a refund?"},
            "severity":  {"type": "score", "criteria": ["trivial", "minor", "moderate", "serious", "critical"]},
        },
        temperature=1.47,
    )

    answer.sentiment.result          # 'negative'
    answer.sentiment.confidence      # 0.99...
    answer.sentiment.probabilities   # {'negative': 0.99, 'positive': 0.01}
    answer.refund.result             # True
    answer.severity.result           # 2.2 (expected level)
    answer["sentiment"]              # same Answer by id

With a single question, the answer is proxied, so `answer.result` works directly.
`questions` may also be a JSON string, and `content` may be a string, dict or list.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.environ.get("KEV_SOURCE") or os.path.join(HERE, "kev-source")
VENV_PY = os.path.join(SOURCE, ".venv", "bin", "python")
RUN = os.environ.get("KEV_RUN") or os.path.join(SOURCE, "runs", "kev")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _bootstrap():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != here]
    if os.path.exists(VENV_PY) and os.path.realpath(sys.executable) != os.path.realpath(VENV_PY):
        os.execv(VENV_PY, [VENV_PY, os.path.abspath(__file__), *sys.argv[1:]])
    if SOURCE not in sys.path:
        sys.path.insert(0, SOURCE)


_bootstrap()

import json
from dataclasses import dataclass
from typing import Any

import torch

from kev.api import SystemOneRequest, choice_confidence, score_confidence, to_record
from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import admit

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass
class Answer:
    id: str
    type: str
    result: Any
    confidence: float
    probabilities: dict

    @classmethod
    def from_probs(cls, meta, probs):
        keys = meta["keys"]
        if meta["type"] == "noul":
            result = bool(probs[1] >= probs[0])
            confidence = choice_confidence(probs)
        elif meta["type"] == "choice":
            result = keys[max(range(len(probs)), key=probs.__getitem__)]
            confidence = choice_confidence(probs)
        else:
            result = sum(i * p for i, p in enumerate(probs))
            confidence = score_confidence(probs)
        return cls(id=meta["id"], type=meta["type"], result=result, confidence=confidence,
                   probabilities=dict(zip(keys, probs)))


class Result(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            if len(self) == 1:
                return getattr(next(iter(self.values())), name)
            raise AttributeError(name)


class Kev:
    def __init__(self, run=RUN, device=None, dtype=None, temperature=None):
        self.checkpoint = Checkpoint(run)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        name = (dtype or os.environ.get("KEV_DTYPE", "fp32")).lower()
        self.tok, self.model = self.checkpoint.load(self.device, LoadOptions(dtype=DTYPES.get(name)))
        self.temperature = temperature if temperature is not None else float(os.environ.get("KEV_TEMPERATURE", self.model.head.temperature))

    def ask(self, content, questions, temperature=None):
        if isinstance(questions, str):
            questions = json.loads(questions)
        req = SystemOneRequest(state=content, questions=questions)
        rec, meta = to_record(req)
        enc = admit(self.model, self.tok, rec)
        t = self.temperature if temperature is None else temperature
        previous = self.model.head.temperature
        self.model.head.temperature = t
        try:
            probs = self.model.probs(enc)
        finally:
            self.model.head.temperature = previous
        return Result((m["id"], Answer.from_probs(m, p.tolist())) for m, p in zip(meta, probs))


def load(**kwargs):
    return Kev(**kwargs)
