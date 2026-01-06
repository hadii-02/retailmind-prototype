# src/run_pipeline.py
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from src.pipeline.io import Paths, ensure_dir
from src.pipeline.llm_cache import JsonFileCache

from src.pipeline.stage_llm_satisfaction import stage_llm_satisfaction
from src.pipeline.stage_issue_tagging import stage_issue_tagging
from src.pipeline.stage_topic_clustering import stage_topic_clustering
from src.pipeline.stage_prompt_repair import stage_prompt_repair

# ---- Load environment variables from .env (repo root) ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# -----------------------------
# Repo helpers
# -----------------------------
def repo_root() -> Path:
    # repo_root/src/run_pipeline.py -> repo_root
    return Path(__file__).resolve().parents[1]


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _require_exists(p: Path, label: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")


def pick_api_key(key_name: str) -> str:
    """
    key_name: environment variable name containing the API key.
    Example: OPENAI_API_KEY or OPENAI_API_KEY1
    """
    k = os.getenv(key_name)
    if not k:
        raise RuntimeError(
            f"{key_name} is not set. Put it in .env (repo root) or set it as an environment variable."
        )
    return k


def get_openai_client(api_key_env: str) -> "OpenAI":
    """
    Create an OpenAI client using the selected environment variable.
    """
    from openai import OpenAI
    api_key = pick_api_key(api_key_env)
    return OpenAI(api_key=api_key)


def _retry_call(fn: Callable[[], Dict[str, Any]], max_retries: int = 5, base_sleep: float = 1.5) -> Dict[str, Any]:
    """
    Conservative retry (rate limits, transient errors).
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (2 ** (attempt - 1))
            sleep_s = min(sleep_s, 20.0)
            print(f"[WARN] LLM call failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(sleep_s)
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


# -----------------------------
# Cache key helpers
# -----------------------------
def _stable_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# -----------------------------
# Pipeline configuration
# -----------------------------
@dataclass(frozen=True)
class DemoPipelineConfig:
    """
    Demo pipeline runs only on the English subset.
    We do NOT commit processed parquet outputs; we regenerate locally.
    """
    # Base input (must exist locally)
    turns_in: Path

    # Stage outputs (local)
    turns_satisfaction_out: Path
    issue_tagged_out: Path

    topics_rows_out: Path
    topics_summary_out: Path
    turns_topics_out: Path

    prompt_repairs_out: Path

    # LLM cache directory
    cache_dir: Path

    # Clustering params
    k_topics: int = 6
    max_cluster_rows: int = 5000

    # Model names
    satisfaction_model: str = "gpt-4o-mini"
    issue_model: str = "gpt-4o-mini"
    repair_model: str = "gpt-4o-mini"


def default_config(paths: Paths) -> DemoPipelineConfig:
    processed = paths.processed_dir

    # Base file you already have locally.
    turns_in = processed / "uss_english_turns_satisfaction.parquet"

    # Consistent naming (matches notebooks)
    turns_satisfaction_out = processed / "uss_english_turns_satisfaction_llm_subset.parquet"
    issue_tagged_out = processed / "uss_english_issue_tagged_llm_subset.parquet"

    topics_rows_out = processed / "uss_english_topics_rows_llm_subset.parquet"
    topics_summary_out = processed / "uss_english_topics_summary_llm_subset.parquet"
    turns_topics_out = processed / "uss_english_turns_topics_llm_subset.parquet"

    prompt_repairs_out = processed / "uss_english_prompt_repairs_llm_subset.parquet"

    cache_dir = processed / "llm_cache"

    return DemoPipelineConfig(
        turns_in=turns_in,
        turns_satisfaction_out=turns_satisfaction_out,
        issue_tagged_out=issue_tagged_out,
        topics_rows_out=topics_rows_out,
        topics_summary_out=topics_summary_out,
        turns_topics_out=turns_topics_out,
        prompt_repairs_out=prompt_repairs_out,
        cache_dir=cache_dir,
        k_topics=int(_env("RM_K_TOPICS", "6") or 6),
        max_cluster_rows=int(_env("RM_MAX_CLUSTER_ROWS", "5000") or 5000),
        satisfaction_model=_env("RM_SAT_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        issue_model=_env("RM_ISSUE_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        repair_model=_env("RM_REPAIR_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
    )


# -----------------------------
# Demo subset helper
# -----------------------------
def make_demo_subset_turns(
    turns_path: Path,
    out_path: Path,
    max_convs: int,
    seed: int = 42,
) -> Path:
    """
    Create a smaller demo subset by sampling conversations (dataset, conv_id).
    Keeps all turns for selected conversations.
    """
    import pandas as pd

    df = pd.read_parquet(turns_path)

    required = {"dataset", "conv_id", "turn_id", "speaker", "text"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"turns parquet missing columns: {missing}")

    convs = df[["dataset", "conv_id"]].drop_duplicates()

    if len(convs) <= max_convs:
        df_sub = df
    else:
        convs_sub = convs.sample(n=max_convs, random_state=seed)
        df_sub = df.merge(convs_sub, on=["dataset", "conv_id"], how="inner")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_sub.to_parquet(out_path, index=False)

    n_convs = df_sub[["dataset", "conv_id"]].drop_duplicates().shape[0]
    print(f"[DEMO] Subset turns saved: {out_path}")
    print(f"       rows={len(df_sub)} | convs={n_convs} | max_convs={max_convs} | seed={seed}")
    return out_path


# -----------------------------
# Cached LLM functions
# -----------------------------
def make_cached_satisfaction_score_fn(
    cache: JsonFileCache,
    model_name: str,
    api_key_env: str,
) -> Callable[[str, str], Dict[str, Any]]:
    """
    score_fn(user_text, last_system_text) -> {"score": int 1..5, "reason": str}
    Cached by (model + normalized texts).
    """
    client = get_openai_client(api_key_env=api_key_env)

    def _score(user_text: str, last_system_text: str) -> Dict[str, Any]:
        u = (user_text or "").strip()
        s = (last_system_text or "").strip()

        key_payload = {
            "task": "satisfaction_score",
            "model": model_name,
            "user_text": u,
            "last_system_text": s,
        }
        key = _stable_json(key_payload)
        cached = cache.get(key)
        if cached is not None:
            return cached

        prompt = f"""
You are scoring user satisfaction for a virtual assistant conversation.

Input:
- last assistant message (context)
- user message

Return a JSON object:
{{
  "score": 1|2|3|4|5,
  "reason": "short justification (1 sentence)"
}}

Assistant (last message):
{s}

User:
{u}
""".strip()

        def _do() -> Dict[str, Any]:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a strict rater. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            return json.loads(resp.choices[0].message.content)

        data = _retry_call(_do)
        cache.set(key, data)
        return data

    return _score


def make_cached_issue_fn(
    cache: JsonFileCache,
    model_name: str,
    api_key_env: str,
) -> Callable[[str], Dict[str, Any]]:
    """
    issue_fn(snippet) -> {"issues": [...], "severity": "...", "reason": "..."}
    Cached by (model + snippet).
    """
    client = get_openai_client(api_key_env=api_key_env)

    ISSUE_LABELS = [
        "MISSING_CONTEXT",
        "WRONG_FACT",
        "TONE_ISSUE",
        "LOOP",
        "UNSUPPORTED_INTENT",
        "SLOW_RESPONSE",
        "HANDOFF_REQUIRED",
        "SUCCESS_BEST_PRACTICE",
    ]

    def _issue(snippet: str) -> Dict[str, Any]:
        sn = (snippet or "").strip()

        key_payload = {
            "task": "issue_tagging",
            "model": model_name,
            "snippet": sn,
        }
        key = _stable_json(key_payload)
        cached = cache.get(key)
        if cached is not None:
            return cached

        taxonomy_str = ", ".join(ISSUE_LABELS)
        prompt = f"""
You are diagnosing why a chatbot reply led to low user satisfaction.

Taxonomy: {taxonomy_str}

Output JSON:
{{
  "issues": ["..."],  # at least one; use SUCCESS_BEST_PRACTICE if no real failure
  "severity": "LOW"|"MEDIUM"|"HIGH",
  "reason": "1-2 sentences"
}}

Snippet:
---
{sn}
---
""".strip()

        def _do() -> Dict[str, Any]:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a failure analyst. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            return json.loads(resp.choices[0].message.content)

        data = _retry_call(_do)
        cache.set(key, data)
        return data

    return _issue


def make_cached_repair_fn(
    cache: JsonFileCache,
    model_name: str,
    api_key_env: str,
) -> Callable[[str], Dict[str, Any]]:
    """
    repair_fn(topic_description) -> dict with:
      root_cause, suggested_prompt_changes, system_prompt_snippet, guardrail_rules, evaluation_checks
    Cached by (model + topic_description).
    """
    client = get_openai_client(api_key_env=api_key_env)

    def _repair(topic_description: str) -> Dict[str, Any]:
        td = (topic_description or "").strip()

        key_payload = {
            "task": "prompt_repair",
            "model": model_name,
            "topic_description": td,
        }
        key = _stable_json(key_payload)
        cached = cache.get(key)
        if cached is not None:
            return cached

        prompt = f"""
You are an expert conversation designer for virtual assistants.

You are given a TOPIC representing a cluster of low-satisfaction user turns.
The topic description includes frequent issue labels, representative messages, and a diagnostic reason.

Return JSON with exactly:
{{
  "root_cause": "string",
  "suggested_prompt_changes": ["string", "..."],
  "system_prompt_snippet": "string",
  "guardrail_rules": ["string", "..."],
  "evaluation_checks": ["string", "..."]
}}

Be specific and practical (developer-facing).

TOPIC:
---
{td}
---
""".strip()

        def _do() -> Dict[str, Any]:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a conversation designer. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            return json.loads(resp.choices[0].message.content)

        data = _retry_call(_do)
        cache.set(key, data)
        return data

    return _repair


# -----------------------------
# Embeddings for clustering (MiniLM local)
# -----------------------------
def make_minilm_embed_fn() -> Callable[[list[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    def _embed(texts: list[str]) -> np.ndarray:
        vecs = model.encode(
            ["" if t is None else str(t) for t in texts],
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True,
        )
        return np.asarray(vecs)

    return _embed


# -----------------------------
# Pipeline orchestration
# -----------------------------
def run_demo_pipeline(
    cfg: DemoPipelineConfig,
    api_key_env: str,
    skip_existing: bool = True,
    from_stage: int = 1,
    to_stage: int = 4,
) -> Dict[str, Path]:
    """
    Runs end-to-end demo pipeline on the English subset.
    Stages:
      1) LLM satisfaction
      2) Issue tagging
      3) Topic clustering
      4) Prompt repair
    """
    ensure_dir(cfg.cache_dir)
    cache = JsonFileCache(cfg.cache_dir)

    _require_exists(cfg.turns_in, "Base English turns parquet")

    artifacts: Dict[str, Path] = {}

    # Stage 1 — LLM satisfaction
    if from_stage <= 1 <= to_stage:
        if (not skip_existing) or (not cfg.turns_satisfaction_out.exists()):
            print("[STAGE 1] LLM satisfaction scoring...")
            score_fn = make_cached_satisfaction_score_fn(
                cache, cfg.satisfaction_model, api_key_env=api_key_env
            )
            stage_llm_satisfaction(
                turns_path=cfg.turns_in,
                out_path=cfg.turns_satisfaction_out,
                score_fn=score_fn,
            )
        else:
            print("[STAGE 1] Skipped (output exists).")
        artifacts["turns_satisfaction"] = cfg.turns_satisfaction_out

    # Stage 2 — Issue tagging
    if from_stage <= 2 <= to_stage:
        _require_exists(cfg.turns_satisfaction_out, "Stage 1 output (turns_satisfaction_out)")
        if (not skip_existing) or (not cfg.issue_tagged_out.exists()):
            print("[STAGE 2] Issue tagging...")
            issue_fn = make_cached_issue_fn(cache, cfg.issue_model, api_key_env=api_key_env)
            stage_issue_tagging(
                turns_satisfaction_path=cfg.turns_satisfaction_out,
                out_path=cfg.issue_tagged_out,
                issue_fn=issue_fn,
            )
        else:
            print("[STAGE 2] Skipped (output exists).")
        artifacts["issue_tagged"] = cfg.issue_tagged_out

    # Stage 3 — Topic clustering
    if from_stage <= 3 <= to_stage:
        _require_exists(cfg.issue_tagged_out, "Stage 2 output (issue_tagged_out)")
        needs = (
            (not cfg.topics_rows_out.exists())
            or (not cfg.topics_summary_out.exists())
            or (not cfg.turns_topics_out.exists())
        )
        if (not skip_existing) or needs:
            print("[STAGE 3] Topic clustering (MiniLM + KMeans)...")
            embed_fn = make_minilm_embed_fn()
            stage_topic_clustering(
                issue_tagged_path=cfg.issue_tagged_out,
                topics_rows_out=cfg.topics_rows_out,
                topics_summary_out=cfg.topics_summary_out,
                turns_topics_out=cfg.turns_topics_out,
                embed_fn=embed_fn,
                k=cfg.k_topics,
                max_rows=cfg.max_cluster_rows,
            )
        else:
            print("[STAGE 3] Skipped (outputs exist).")
        artifacts["topics_rows"] = cfg.topics_rows_out
        artifacts["topics_summary"] = cfg.topics_summary_out
        artifacts["turns_topics"] = cfg.turns_topics_out

    # Stage 4 — Prompt repair
    if from_stage <= 4 <= to_stage:
        _require_exists(cfg.topics_summary_out, "Stage 3 output (topics_summary_out)")
        _require_exists(cfg.turns_topics_out, "Stage 3 output (turns_topics_out)")
        if (not skip_existing) or (not cfg.prompt_repairs_out.exists()):
            print("[STAGE 4] Prompt repair generation...")
            repair_fn = make_cached_repair_fn(cache, cfg.repair_model, api_key_env=api_key_env)
            stage_prompt_repair(
                topics_summary_path=cfg.topics_summary_out,
                turns_topics_path=cfg.turns_topics_out,
                repairs_out_path=cfg.prompt_repairs_out,
                repair_fn=repair_fn,
            )
        else:
            print("[STAGE 4] Skipped (output exists).")
        artifacts["prompt_repairs"] = cfg.prompt_repairs_out

    print("\n[DONE] Demo pipeline completed.")
    for k, p in artifacts.items():
        print(f" - {k}: {p}")
    print(f" - llm_cache_dir: {cfg.cache_dir}")

    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="RetailMind demo pipeline runner (English subset only).")

    # Core controls
    parser.add_argument("--no-skip", action="store_true", help="Recompute stages even if outputs exist.")
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY1",
        help="Env var containing OpenAI API key (default: OPENAI_API_KEY1 for dev).",
    )

    # Demo subset controls
    parser.add_argument(
        "--demo-max-convs",
        type=int,
        default=80,
        help="Cap number of conversations for demo by sampling conversations.",
    )
    parser.add_argument("--demo-seed", type=int, default=42, help="Random seed for demo sampling.")
    parser.add_argument(
        "--no-demo-subset",
        action="store_true",
        help="Disable demo subset and run on full turns_in parquet.",
    )

    # Stage range controls
    parser.add_argument("--from-stage", type=int, default=1, choices=[1, 2, 3, 4], help="Start from stage N.")
    parser.add_argument("--to-stage", type=int, default=4, choices=[1, 2, 3, 4], help="Run up to stage N.")

    # Optional overrides
    parser.add_argument("--turns-in", type=str, default=None, help="Override base English turns parquet path.")
    parser.add_argument("--k", type=int, default=None, help="Override number of KMeans topics.")
    parser.add_argument("--max-rows", type=int, default=None, help="Override max rows used for clustering.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override cache directory path.")

    args = parser.parse_args()

    root = repo_root()
    paths = Paths.from_repo_root(root)
    ensure_dir(paths.processed_dir)

    cfg = default_config(paths)

    # Apply CLI overrides
    if args.turns_in is not None:
        cfg = DemoPipelineConfig(**{**cfg.__dict__, "turns_in": Path(args.turns_in)})
    if args.k is not None:
        cfg = DemoPipelineConfig(**{**cfg.__dict__, "k_topics": int(args.k)})
    if args.max_rows is not None:
        cfg = DemoPipelineConfig(**{**cfg.__dict__, "max_cluster_rows": int(args.max_rows)})
    if args.cache_dir is not None:
        cfg = DemoPipelineConfig(**{**cfg.__dict__, "cache_dir": Path(args.cache_dir)})

    # Build demo subset turns parquet (recommended for live demo)
    if not args.no_demo_subset:
        demo_subset_path = paths.processed_dir / "uss_english_turns_demo_subset.parquet"
        cfg = DemoPipelineConfig(**{
            **cfg.__dict__,
            "turns_in": make_demo_subset_turns(
                turns_path=cfg.turns_in,
                out_path=demo_subset_path,
                max_convs=args.demo_max_convs,
                seed=args.demo_seed,
            )
        })

    run_demo_pipeline(
        cfg,
        api_key_env=args.api_key_env,
        skip_existing=(not args.no_skip),
        from_stage=args.from_stage,
        to_stage=args.to_stage,
    )


if __name__ == "__main__":
    main()
