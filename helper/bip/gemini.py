"""Gemini on Vertex AI, with the service account kept in a .env file (never printed or copied).

    generate(cfg, text, images=[jpeg bytes], schema={...}) -> (reply text, usage dict, latency s)
"""

import json
import time
from functools import cache
from pathlib import Path


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


def generate(cfg, text: str, images: list[bytes] = (), schema: dict | None = None) -> tuple[str, dict, float]:
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
