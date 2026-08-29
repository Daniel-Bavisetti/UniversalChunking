"""Model providers.

One OpenAI-compatible adapter covers every vendor we care about, because OpenAI,
Gemini, DashScope, Ollama, vLLM and OpenRouter all speak the same
chat-completions schema. Pointing `base_url` somewhere else is the entire
"multi-provider support" story, so there is no adapter tree here.

The offline implementations are the DEFAULT and are always available. They are
deliberately honest: the offline vision provider reports what it measured and
never invents a scene description.

Configure with environment variables:

    VKE_LLM_PROVIDER=openai|offline      (default: offline)
    VKE_VISION_PROVIDER=openai|offline   (default: offline)
    VKE_API_KEY=sk-...
    VKE_BASE_URL=https://api.openai.com/v1
        # Gemini:  https://generativelanguage.googleapis.com/v1beta/openai/
        # Ollama:  http://localhost:11434/v1
    VKE_LLM_MODEL=gpt-4o-mini
    VKE_VISION_MODEL=gpt-4o-mini

Copy .env.example to .env and fill in values; it is loaded automatically (and
gitignored, so real keys never get committed). Actual environment variables
always take precedence over .env, so CI/deploy configs still work unchanged.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .schemas import VisionResult

try:
    from dotenv import load_dotenv

    # override=False: real env vars set by the shell/deploy environment win.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv is optional; plain env vars still work without it

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProviderSettings:
    llm: str = field(default_factory=lambda: os.getenv("VKE_LLM_PROVIDER", "offline"))
    vision: str = field(default_factory=lambda: os.getenv("VKE_VISION_PROVIDER", "offline"))
    api_key: str = field(default_factory=lambda: os.getenv("VKE_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.getenv("VKE_BASE_URL", "https://api.openai.com/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("VKE_LLM_MODEL", "gpt-4o-mini"))
    vision_model: str = field(
        default_factory=lambda: os.getenv("VKE_VISION_MODEL", "gpt-4o-mini"))
    max_vision_calls: int = field(
        default_factory=lambda: int(os.getenv("VKE_MAX_VISION_CALLS", "40")))


# --------------------------------------------------------------------------- #
# interfaces
# --------------------------------------------------------------------------- #
class LLM(Protocol):
    name: str

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 400) -> str: ...


class Vision(Protocol):
    name: str

    def describe(self, image_path: Path, transcript: str) -> VisionResult: ...


# --------------------------------------------------------------------------- #
# usage accounting - the efficiency claim has to be measurable
# --------------------------------------------------------------------------- #
@dataclass
class Usage:
    llm_calls: int = 0
    vision_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failures: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "llm_calls": self.llm_calls,
            "vision_calls": self.vision_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "failures": self.failures,
        }


USAGE = Usage()


# --------------------------------------------------------------------------- #
# offline (default)
# --------------------------------------------------------------------------- #
class OfflineLLM:
    """No model. Callers fall back to their own extractive logic."""

    name = "offline"
    available = True

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 400) -> str:
        return ""


class OfflineVision:
    """Measured descriptors only.

    Inventing a scene description with no model behind it would be dishonest, so
    this returns nothing for the semantic fields and lets the keyframe image and
    the measured numbers speak.
    """

    name = "offline"
    available = True

    def describe(self, image_path: Path, transcript: str) -> VisionResult:
        return VisionResult(source="heuristic")


# --------------------------------------------------------------------------- #
# OpenAI-compatible (OpenAI, Gemini, DashScope, Ollama, vLLM, OpenRouter)
# --------------------------------------------------------------------------- #
def _client(settings: ProviderSettings):
    from openai import OpenAI

    return OpenAI(api_key=settings.api_key or "not-needed",
                  base_url=settings.base_url)


class OpenAICompatLLM:
    name = "openai-compat"

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.name = f"openai-compat:{settings.llm_model}"
        self._client = _client(settings)

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 400) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self._client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            USAGE.llm_calls += 1
            if resp.usage:
                USAGE.prompt_tokens += resp.usage.prompt_tokens or 0
                USAGE.completion_tokens += resp.usage.completion_tokens or 0
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # a provider outage must not kill the run
            USAGE.failures += 1
            print(f"[providers] LLM call failed ({type(exc).__name__}: {exc})")
            return ""


VISION_PROMPT = """You are analysing ONE frame from a video.

Return STRICT JSON with exactly these keys:
{
  "description": "one factual sentence about what is visible",
  "ocr_text": ["text visibly rendered on screen, verbatim, one string per line"],
  "objects": ["concrete visible objects"],
  "actions": ["what a person appears to be doing, if anyone is visible"]
}

Report only what is actually visible. Use empty lists where nothing applies.
Do not infer from the transcript; it is context only.

Transcript around this moment:
{transcript}"""


class OpenAICompatVision:
    """One call per keyframe returns description, on-screen text, objects and actions.

    Asking a VLM to read on-screen text in the same request is why there is no
    separate OCR provider: a second call would double the cost for the same pixels.
    """

    name = "openai-compat"

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.name = f"openai-compat:{settings.vision_model}"
        self._client = _client(settings)

    def describe(self, image_path: Path, transcript: str) -> VisionResult:
        if USAGE.vision_calls >= self.settings.max_vision_calls:
            return VisionResult(source="budget_exceeded")
        try:
            b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            prompt = VISION_PROMPT.replace("{transcript}", transcript[:600] or "(none)")
            resp = self._client.chat.completions.create(
                model=self.settings.vision_model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]}],
                max_tokens=400,
                temperature=0.1,
            )
            USAGE.vision_calls += 1
            if resp.usage:
                USAGE.prompt_tokens += resp.usage.prompt_tokens or 0
                USAGE.completion_tokens += resp.usage.completion_tokens or 0
            return _parse_vision(resp.choices[0].message.content or "")
        except Exception as exc:
            USAGE.failures += 1
            print(f"[providers] vision call failed ({type(exc).__name__}: {exc})")
            return VisionResult(source="failed")


def _parse_vision(raw: str) -> VisionResult:
    """Tolerate fenced code blocks and stray prose around the JSON."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    else:
        brace = re.search(r"\{.*\}", text, re.S)
        if brace:
            text = brace.group(0)
    try:
        blob = json.loads(text)
    except json.JSONDecodeError:
        # A model that ignored the format still gave us usable prose.
        return VisionResult(description=raw.strip()[:300], source="vlm")

    def strlist(key: str) -> list[str]:
        val = blob.get(key) or []
        if isinstance(val, str):
            val = [val]
        return [str(v).strip() for v in val if str(v).strip()][:12]

    return VisionResult(
        description=str(blob.get("description", "")).strip()[:400],
        ocr_text=strlist("ocr_text"),
        objects=strlist("objects"),
        actions=strlist("actions"),
        source="vlm",
    )


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def get_llm(settings: ProviderSettings | None = None) -> LLM:
    s = settings or ProviderSettings()
    if s.llm == "offline":
        return OfflineLLM()
    if not s.api_key:
        print("[providers] VKE_LLM_PROVIDER is set but VKE_API_KEY is empty; "
              "falling back to offline")
        return OfflineLLM()
    try:
        return OpenAICompatLLM(s)
    except Exception as exc:
        print(f"[providers] could not init LLM ({exc}); falling back to offline")
        return OfflineLLM()


def get_vision(settings: ProviderSettings | None = None) -> Vision:
    s = settings or ProviderSettings()
    if s.vision == "offline":
        return OfflineVision()
    if not s.api_key:
        print("[providers] VKE_VISION_PROVIDER is set but VKE_API_KEY is empty; "
              "falling back to offline")
        return OfflineVision()
    try:
        return OpenAICompatVision(s)
    except Exception as exc:
        print(f"[providers] could not init vision ({exc}); falling back to offline")
        return OfflineVision()


def describe_providers(settings: ProviderSettings | None = None) -> dict[str, str]:
    s = settings or ProviderSettings()
    return {
        "llm": get_llm(s).name,
        "vision": get_vision(s).name,
        "base_url": s.base_url if s.llm != "offline" or s.vision != "offline" else "-",
    }
