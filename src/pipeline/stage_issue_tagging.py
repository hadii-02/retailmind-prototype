# src/pipeline/stage_issue_tagging.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Any

import numpy as np
import pandas as pd


def normalize_list_field(val) -> list[str]:
    """
    Normalize list-like fields for JSON/parquet consistency.
    Handles: list, numpy.ndarray, string, NaN, None.
    """
    if val is None:
        return []
    if isinstance(val, float) and np.isnan(val):
        return []
    if val is pd.NA:
        return []
    if isinstance(val, np.ndarray):
        return [str(x) for x in val.tolist()]
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    return [str(val)]


def stage_issue_tagging(
    turns_satisfaction_path: Path,
    out_path: Path,
    issue_fn: Callable[[str], Dict[str, Any]],
) -> Path:
    """
    Tags issues for low-satisfaction USER turns.
    Produces a full-turn parquet with columns:
      issues (list[str]), severity (str), reason (str)
    """
    df = pd.read_parquet(turns_satisfaction_path)

    required = {"dataset", "conv_id", "turn_id", "speaker", "text", "low_satisfaction"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input satisfaction parquet missing columns: {missing}")

    # Ensure last_system_text exists; if not, reconstruct
    if "last_system_text" not in df.columns:
        df = df.sort_values(["dataset", "conv_id", "turn_id"]).copy()
        df["system_text_only"] = df["text"].where(df["speaker"] == "SYSTEM")
        df["last_system_text"] = df.groupby(["dataset", "conv_id"])["system_text_only"].ffill()

    df_low = df[(df["speaker"] == "USER") & (df["low_satisfaction"] == True)].copy()

    df_low["last_system_text"] = df_low["last_system_text"].fillna("[No previous system turn available]")
    df_low["snippet"] = "Bot: " + df_low["last_system_text"].astype(str) + "\nUser: " + df_low["text"].astype(str)

    tagged_rows = []
    for r in df_low.itertuples(index=False):
        snippet = "" if r.snippet is None else str(r.snippet)
        res = issue_fn(snippet) or {}

        issues = normalize_list_field(res.get("issues", []))
        severity = str(res.get("severity", "MEDIUM") or "MEDIUM").upper().strip()
        reason = str(res.get("reason", "") or "").strip()

        tagged_rows.append({
            "dataset": r.dataset,
            "conv_id": int(r.conv_id),
            "turn_id": int(r.turn_id),
            "issues": issues,
            "severity": severity,
            "reason": reason,
        })

    df_issues = pd.DataFrame(tagged_rows)

    # Merge back to all turns
    df_tagged = df.merge(df_issues, on=["dataset", "conv_id", "turn_id"], how="left")

    # Fill defaults for non-tagged rows
    if "issues" not in df_tagged.columns:
        df_tagged["issues"] = [[] for _ in range(len(df_tagged))]
    else:
        df_tagged["issues"] = df_tagged["issues"].apply(normalize_list_field)

    df_tagged["severity"] = df_tagged["severity"].fillna("NONE")
    df_tagged["reason"] = df_tagged["reason"].fillna("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_tagged.to_parquet(out_path, index=False)
    return out_path
