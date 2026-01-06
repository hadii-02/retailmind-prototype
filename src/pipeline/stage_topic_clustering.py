# src/pipeline/stage_topic_clustering.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

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


def stage_topic_clustering(
    issue_tagged_path: Path,
    topics_rows_out: Path,
    topics_summary_out: Path,
    turns_topics_out: Path,
    embed_fn: Callable[[list[str]], np.ndarray],
    k: int = 6,
    max_rows: int = 5000,
) -> Tuple[Path, Path, Path]:
    df = pd.read_parquet(issue_tagged_path)

    required = {"dataset", "conv_id", "turn_id", "speaker", "text", "low_satisfaction", "issues"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input issue-tagged parquet missing columns: {missing}")

    def has_real_issue(x) -> bool:
        lst = normalize_list_field(x)
        if not lst:
            return False
        return any(i != "SUCCESS_BEST_PRACTICE" for i in lst)

    mask = (
        (df["speaker"] == "USER")
        & (df["low_satisfaction"] == True)
        & (df["issues"].apply(has_real_issue))
    )

    df_cluster = df.loc[mask, ["dataset","conv_id","turn_id","text","issues","severity","reason"]].copy()
    if len(df_cluster) == 0:
        raise RuntimeError("No rows selected for clustering. Check low_satisfaction/issues logic.")

    if len(df_cluster) > max_rows:
        df_cluster = df_cluster.sample(max_rows, random_state=42).copy()

    texts = df_cluster["text"].astype(str).tolist()
    X = embed_fn(texts)
    X = np.asarray(X)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=int(k), random_state=42, n_init="auto")
    df_cluster["topic_id"] = km.fit_predict(X).astype(int)

    df_cluster["issues_norm"] = df_cluster["issues"].apply(normalize_list_field)

    # Build topic labels using top issue labels
    from collections import Counter
    summaries = []
    for tid, g in df_cluster.groupby("topic_id"):
        c = Counter()
        for lst in g["issues_norm"]:
            for it in lst:
                c[it] += 1
        top_issues = [x for x, _ in c.most_common(3)]
        label_core = " / ".join([x for x, _ in c.most_common(2)]) or "General"

        example_texts = g["text"].head(3).astype(str).tolist()
        reasons = [r for r in g["reason"].astype(str).tolist() if r and r.lower() != "nan"]
        example_reason = reasons[0] if reasons else ""

        summaries.append({
            "topic_id": int(tid),
            "topic_label": f"Topic {int(tid)}: {label_core}",
            "n_examples": int(len(g)),
            "top_issues": top_issues,
            "example_texts": example_texts,
            "example_reason": example_reason,
        })

    df_topics_summary = pd.DataFrame(summaries).sort_values("n_examples", ascending=False)

    df_topics_rows = df_cluster[["dataset","conv_id","turn_id","topic_id","text"]].copy()

    # Attach topic_id / topic_label to all turns
    df_turns_topics = df.merge(
        df_topics_rows[["dataset","conv_id","turn_id","topic_id"]],
        on=["dataset","conv_id","turn_id"],
        how="left",
    )
    df_turns_topics["topic_id"] = df_turns_topics["topic_id"].fillna(-1).astype(int)
    label_map = df_topics_summary.set_index("topic_id")["topic_label"].to_dict()
    df_turns_topics["topic_label"] = df_turns_topics["topic_id"].map(label_map).fillna("UNCLUSTERED")

    topics_rows_out.parent.mkdir(parents=True, exist_ok=True)
    df_topics_rows.to_parquet(topics_rows_out, index=False)
    df_topics_summary.to_parquet(topics_summary_out, index=False)
    df_turns_topics.to_parquet(turns_topics_out, index=False)

    return topics_rows_out, topics_summary_out, turns_topics_out
