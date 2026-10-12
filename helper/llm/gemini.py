"""Gemini on Vertex AI, with the service account kept in a .env file (never printed or copied).

The .env holds GOOGLE_CREDENTIALS_JSON (the service-account JSON, multi-line), GOOGLE_CLOUD_PROJECT and
GOOGLE_CLOUD_LOCATION. Models that answer on this project: gemini-3.5-flash, gemini-3.6-flash,
gemini-3.1-pro-preview.

    from helper.llm.gemini import Gemini, generate
    text, usage, latency = generate("Reply with {\"ok\": true}", schema={"type": "object"})
    text, usage, latency = generate("What is in view?", images=[jpeg_bytes], cfg=Gemini(model="gemini-3.6-flash"))
"""

import json
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Gemini:
    model: str = "gemini-3.5-flash"
    env: str = str(ROOT.parent / "MiniWeb-src/MiniWeb/.env")  # service account for Vertex AI
    thinking: str = "low"  # thinking level, "" for the model default
    temperature: float = 0.0
    max_tokens: int = 8192


def env(path: str, name: str) -> str:
    """One value of a .env file, including multi-line quoted values such as GOOGLE_CREDENTIALS_JSON."""
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines):
        s = line.strip().removeprefix("export ")
        if not s.startswith(name + "="):
            continue
        raw = s.split("=", 1)[1]
        if raw.startswith('"') and not (len(raw) > 1 and raw.rstrip().endswith('"')):
            parts = [raw[1:]]
            for cont in lines[i + 1:]:
                if cont.rstrip().endswith('"') and not cont.rstrip().endswith('\\"'):
                    parts.append(cont.rstrip()[:-1])
                    break
                parts.append(cont)
            return "\n".join(parts).replace('\\"', '"').replace("\\\\", "\\")
        return raw.strip().strip('"').strip("'")
    return ""


@cache
def client(path: str):
    from google import genai
    from google.oauth2 import service_account
    creds = service_account.Credentials.from_service_account_info(
        json.loads(env(path, "GOOGLE_CREDENTIALS_JSON")), scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return genai.Client(vertexai=True, credentials=creds, project=env(path, "GOOGLE_CLOUD_PROJECT"),
                        location=env(path, "GOOGLE_CLOUD_LOCATION") or "global")


def generate(text: str, images: list[bytes] = (), schema: dict | None = None, cfg: Gemini = Gemini()) -> tuple[str, dict, float]:
    """Reply text, token usage and latency (s). With `schema`, the reply is JSON that follows it.
    Network errors and rate limits are retried with backoff; after four failures the reply says so."""
    from google.genai import types
    conf = types.GenerateContentConfig(
        temperature=cfg.temperature, max_output_tokens=cfg.max_tokens,
        response_mime_type="application/json" if schema else None, response_json_schema=schema,
        thinking_config=types.ThinkingConfig(thinking_level=cfg.thinking) if cfg.thinking else None)
    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in images] + [text]
    for attempt in range(4):
        start = time.perf_counter()
        try:
            r = client(cfg.env).models.generate_content(model=cfg.model, contents=parts, config=conf)
            um = r.usage_metadata
            usage = {"prompt_tokens": um.prompt_token_count or 0, "completion_tokens": um.candidates_token_count or 0,
                     "thinking_tokens": um.thoughts_token_count or 0}
            return r.text or "", usage, time.perf_counter() - start
        except Exception as e:  # network errors and rate limits: back off and retry
            if attempt == 3:
                return f"request failed: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0}, \
                    time.perf_counter() - start
            time.sleep(5 * 2 ** attempt)
