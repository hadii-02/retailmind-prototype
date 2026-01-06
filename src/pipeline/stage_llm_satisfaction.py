# src/pipeline/stage_llm_satisfaction.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Any

import numpy as np
import pandas as pd


def add_last_system_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct last SYSTEM turn in each conversation and attach it to all following turns.
    """
    df_sorted = df.sort_values(["dataset", "conv_id", "turn_id"]).copy()
    df_sorted["system_text_only"] = df_sorted["text"].where(df_sorted["speaker"] == "SYSTEM")
    df_sorted["last_system_text"] = (
        df_sorted.groupby(["dataset", "conv_id"])["system_text_only"].ffill()
    )
    return df_sorted


def stage_llm_satisfaction(
    turns_path: Path,
    out_path: Path,
    score_fn: Callable[[str, str], Dict[str, Any]],
) -> Path:
    """
    Loads turns parquet, computes LLM satisfaction for USER turns (non-overall),
    and saves a parquet with:
      - last_system_text
      - satisfaction_score (int)
      - satisfaction_reason (str)
      - satisfaction_source (str)
      - low_satisfaction (bool)
    """
    df = pd.read_parquet(turns_path)

    # Ensure required columns exist
    required = {"dataset", "conv_id", "turn_id", "speaker", "text"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input turns parquet missing columns: {missing}")

    df = add_last_system_text(df)

    # Default is_overall if not present
    if "is_overall" not in df.columns:
        df["is_overall"] = False

    mask_score = (df["speaker"] == "USER") & (df["is_overall"] == False)
    df_score = df.loc[mask_score, ["dataset", "conv_id", "turn_id", "text", "last_system_text"]].copy()

    results = []
    for r in df_score.itertuples(index=False):
        user_text = "" if r.text is None else str(r.text)
        last_sys = "" if r.last_system_text is None else str(r.last_system_text)
        out = score_fn(user_text, last_sys)

        # Defensive normalization
        score = out.get("score", None)
        try:
            score = int(score)
        except Exception:
            score = None

        reason = str(out.get("reason", "") or "").strip()

        results.append({
            "dataset": r.dataset,
            "conv_id": int(r.conv_id),
            "turn_id": int(r.turn_id),
            "satisfaction_score": score,
            "satisfaction_reason": reason,
            "satisfaction_source": "llm",
        })

    df_scores = pd.DataFrame(results)

    df = df.merge(df_scores, on=["dataset", "conv_id", "turn_id"], how="left")

    # low_satisfaction only meaningful for USER turns; leave False otherwise
    df["low_satisfaction"] = False
    user_scores = pd.to_numeric(df["satisfaction_score"], errors="coerce")
    df.loc[df["speaker"] == "USER", "low_satisfaction"] = (user_scores < 3).fillna(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path
