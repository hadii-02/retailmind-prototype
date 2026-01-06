# src/pipeline/stage_prompt_repair.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Any

import numpy as np
import pandas as pd


def normalize_list_field(val) -> list[str]:
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


def build_topic_description_from_summary_row(
    summary_row: pd.Series,
    extra_examples: list[str] | None = None,
    max_examples: int = 5,
) -> str:
    topic_id = int(summary_row["topic_id"])
    topic_label = str(summary_row.get("topic_label", "") or "").strip()
    n_examples = int(summary_row.get("n_examples", 0))

    top_issues = normalize_list_field(summary_row.get("top_issues", []))
    top_issues_str = ", ".join(top_issues) if top_issues else "[none]"

    example_texts = normalize_list_field(summary_row.get("example_texts", []))
    if extra_examples:
        example_texts.extend([str(x) for x in extra_examples])
    example_texts = example_texts[:max_examples]

    example_reason = summary_row.get("example_reason", "")
    if example_reason is None or (isinstance(example_reason, float) and np.isnan(example_reason)):
        example_reason = ""
    example_reason = str(example_reason).strip()

    lines = []
    lines.append(f"Topic ID: {topic_id}")
    if topic_label:
        lines.append(f"Topic label: {topic_label}")
    lines.append(f"Number of examples: {n_examples}")
    lines.append(f"Most frequent issue labels: {top_issues_str}")
    lines.append("")
    lines.append("Representative user messages:")
    if example_texts:
        for i, txt in enumerate(example_texts, start=1):
            lines.append(f"{i}) {txt}")
    else:
        lines.append("[No example texts available]")

    if example_reason:
        lines.append("")
        lines.append("Representative diagnostic reason:")
        lines.append(example_reason)

    return "\n".join(lines)


def stage_prompt_repair(
    topics_summary_path: Path,
    turns_topics_path: Path,
    repairs_out_path: Path,
    repair_fn: Callable[[str], Dict[str, Any]],
) -> Path:
    df_topics_summary = pd.read_parquet(topics_summary_path)
    df_turns = pd.read_parquet(turns_topics_path)

    required = {"topic_id", "n_examples", "top_issues", "example_texts", "example_reason"}
    missing = [c for c in required if c not in df_topics_summary.columns]
    if missing:
        raise ValueError(f"topics summary missing columns: {missing}")

    repairs = []
    for _, row in df_topics_summary.iterrows():
        tid = int(row["topic_id"])

        extra_examples = (
            df_turns[(df_turns["topic_id"] == tid) & (df_turns["speaker"] == "USER")]["text"]
            .head(3)
            .astype(str)
            .tolist()
        )

        topic_desc = build_topic_description_from_summary_row(row, extra_examples=extra_examples)
        result = repair_fn(topic_desc) or {}

        # Normalize expected keys (runner will enforce stricter schema later)
        out = {
            "root_cause": str(result.get("root_cause", "") or "").strip(),
            "suggested_prompt_changes": normalize_list_field(result.get("suggested_prompt_changes", [])),
            "system_prompt_snippet": str(result.get("system_prompt_snippet", "") or "").strip(),
            "guardrail_rules": normalize_list_field(result.get("guardrail_rules", [])),
            "evaluation_checks": normalize_list_field(result.get("evaluation_checks", [])),
        }

        repairs.append({
            "topic_id": tid,
            "topic_label": str(row.get("topic_label", f"Topic {tid}")),
            **out,
        })

    df_repairs = pd.DataFrame(repairs)
    repairs_out_path.parent.mkdir(parents=True, exist_ok=True)
    df_repairs.to_parquet(repairs_out_path, index=False)
    return repairs_out_path
