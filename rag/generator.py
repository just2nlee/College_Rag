"""Context assembly and LLM answer generation.

Builds a prompt from the top-k retrieved records and sends it to an LLM.
Supported backends (tried in order):
1. Ollama (local) — llama3 or mistral
2. HuggingFace Inference API (free tier)
3. OpenAI (GPT-4o-mini, if OPENAI_API_KEY is set)

If no LLM backend is available the module returns the assembled context
without a generated answer, so the pipeline still works in "retrieval-only"
mode.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Load .env from project root so API keys are available
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

MAX_CONTEXT_TOKENS = 2048  # rough character budget ≈ tokens * 4
_CHAR_BUDGET = MAX_CONTEXT_TOKENS * 4

# ── context assembly ─────────────────────────────────────────────────

_RECORD_TEMPLATE = textwrap.dedent("""\
    [{course_code}] {title}
    Department: {department}
    Source: {source}
    Description: {description}
    Prerequisites: {prerequisites}
    ---""")


def assemble_context(records: List[Dict[str, Any]]) -> str:
    """Build a textual context block from retrieved records.

    Descriptions are truncated if the total exceeds MAX_CONTEXT_TOKENS.
    """
    blocks: List[str] = []
    char_count = 0
    for rec in records:
        desc = (rec.get("description") or "")
        prereq = rec.get("prerequisites") or "None listed"
        block = _RECORD_TEMPLATE.format(
            course_code=rec.get("course_code", ""),
            title=rec.get("title", ""),
            department=rec.get("department", ""),
            source=rec.get("source", ""),
            description=desc,
            prerequisites=prereq,
        )
        # Truncate if we're running out of budget
        if char_count + len(block) > _CHAR_BUDGET:
            remaining = _CHAR_BUDGET - char_count
            if remaining > 80:
                block = block[:remaining] + "\n[…truncated…]\n---"
            else:
                break
        blocks.append(block)
        char_count += len(block)
    return "\n".join(blocks)


_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a helpful academic advisor for Brown University.
    Answer the user's question using ONLY the course information provided below.
    Cite course codes (e.g. CSCI0150) when referencing specific courses.
    If the answer is not in the provided context, say so clearly.
    Be concise and factual.""")


def build_prompt(query: str, context: str) -> List[Dict[str, str]]:
    """Return a chat-style message list ready for an LLM."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]


# ── LLM backends ────────────────────────────────────────────────────

def _try_ollama(messages: List[Dict[str, str]]) -> Optional[str]:
    """Call a local Ollama instance if available."""
    try:
        import requests

        # Try common Ollama models in order
        for model in ("llama3", "mistral", "llama2"):
            try:
                resp = requests.post(
                    "http://localhost:11434/api/chat",
                    json={"model": model, "messages": messages, "stream": False},
                    timeout=60,
                )
                if resp.ok:
                    data = resp.json()
                    answer = data.get("message", {}).get("content", "")
                    if answer:
                        logger.info("Ollama (%s) generated answer", model)
                        return answer
            except Exception:
                continue
    except ImportError:
        pass
    return None


def _try_huggingface(messages: List[Dict[str, str]]) -> Optional[str]:
    """Call HuggingFace Inference API (free tier)."""
    api_key = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if not api_key:
        return None
    try:
        import requests

        # Use a small chat model available on free tier
        url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        prompt = "\n".join(
            f"{'<s>' if m['role'] == 'system' else ''}[INST] {m['content']} [/INST]"
            if m["role"] != "assistant" else m["content"]
            for m in messages
        )
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 512}},
            timeout=60,
        )
        if resp.ok:
            data = resp.json()
            if isinstance(data, list) and data:
                answer = data[0].get("generated_text", "")
                if answer:
                    logger.info("HuggingFace API generated answer")
                    return answer
    except Exception as exc:
        logger.debug("HuggingFace API failed: %s", exc)
    return None


def _try_openai(messages: List[Dict[str, str]]) -> Optional[str]:
    """Call OpenAI API if OPENAI_API_KEY is set."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import requests

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.3,
            },
            timeout=60,
        )
        if resp.ok:
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            logger.info("OpenAI generated answer")
            return answer
    except Exception as exc:
        logger.debug("OpenAI API failed: %s", exc)
    return None


# ── public API ───────────────────────────────────────────────────────

def generate_answer(
    query: str,
    retrieved_records: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """Assemble context from *retrieved_records* and generate an LLM answer.

    Returns ``(answer, context)`` where *answer* may be a fallback message
    if no LLM backend is available.
    """
    context = assemble_context(retrieved_records)
    messages = build_prompt(query, context)

    # Try backends in priority order
    for backend in (_try_ollama, _try_huggingface, _try_openai):
        answer = backend(messages)
        if answer:
            return answer, context

    # Fallback: no LLM available
    logger.warning(
        "No LLM backend available (Ollama / HuggingFace / OpenAI). "
        "Returning retrieval-only results."
    )
    fallback = (
        "No LLM backend is currently available.  "
        "Here are the most relevant courses found:\n\n" + context
    )
    return fallback, context
