import pandas as pd
from pathlib import Path
from typing import Optional, List

def _parse_scores(s: str) -> Optional[List[int]]:
    """Convert '3,3,3,3' -> [3,3,3,3]."""
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return [int(x) for x in s.split(",")]
    except:
        return None


def load_mwoz(path: str | Path) -> pd.DataFrame:
    """
    Parse MWOZ.txt.
    Format:
    SPEAKER \t TEXT \t DIALOG_ACT(optional) \t SCORES(optional)
    """

    path = Path(path)
    rows = []
    conv_id = 0
    turn_id = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # new conversation
            if not line.strip():
                if turn_id > 0:
                    conv_id += 1
                    turn_id = 0
                continue

            parts = line.split("\t")
            while len(parts) < 4:
                parts.append("")

            speaker, text, act, scores_raw = parts[:4]

            turn_id += 1

            rows.append({
                "conv_id": conv_id,
                "turn_id": turn_id,
                "speaker": speaker,
                "text": text,
                "dialog_act": act if act else None,
                "scores_raw": scores_raw if scores_raw else None,
                "scores": _parse_scores(scores_raw),
                "is_overall": (speaker == "USER" and text == "OVERALL"),
            })

    return pd.DataFrame(rows)


def quick_stats(df):
    print("Rows:", len(df))
    print("Conversations:", df["conv_id"].nunique())
    print("Speakers:", df["speaker"].value_counts().to_dict())
    print("OVERALL lines:", df["is_overall"].sum())
