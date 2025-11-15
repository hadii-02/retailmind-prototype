import json
from pathlib import Path
from typing import Optional, List

import pandas as pd


def _parse_scores(s: str) -> Optional[List[int]]:
    """Convert '3,3,3' -> [3, 3, 3]. Empty -> None."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return [int(x) for x in s.split(",")]
    except ValueError:
        # If something weird appears, keep raw but return None parsed
        return None


def load_sgd(path: str | Path) -> pd.DataFrame:
    """
    Parse the SGD.txt file into a structured DataFrame.

    Columns:
      - conv_id: int         # conversation index
      - turn_id: int         # turn index inside conversation
      - speaker: str         # 'USER' or 'SYSTEM'
      - text: str            # utterance text
      - dialog_act: str|NaN
      - scores_raw: str|NaN  # original '3,3,3' string
      - scores: list[int]    # parsed scores or None
      - is_overall: bool     # True for USER OVERALL lines
    """
    path = Path(path)

    rows = []
    conv_id = 0      # start conversations from 0 (can change to 1 if you prefer)
    turn_id = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # blank line = new conversation
            if not line.strip():
                if turn_id > 0:
                    conv_id += 1
                    turn_id = 0
                continue

            parts = line.split("\t")

            # normalize to 4 columns: speaker, text, act, scores
            while len(parts) < 4:
                parts.append("")
            speaker, text, act, scores_str = parts[:4]

            turn_id += 1

            scores_raw = scores_str.strip() if scores_str else ""
            scores = _parse_scores(scores_raw)

            rows.append(
                {
                    "conv_id": conv_id,
                    "turn_id": turn_id,
                    "speaker": speaker,
                    "text": text,
                    "dialog_act": act if act else None,
                    "scores_raw": scores_raw if scores_raw else None,
                    "scores": scores,
                    "is_overall": (speaker == "USER" and text == "OVERALL"),
                }
            )

    df = pd.DataFrame(rows)
    return df

def quick_stats(df):
    print("Num rows:", len(df))
    print("Num conversations:", df["conv_id"].nunique())
    print("Num OVERALL lines:", df["is_overall"].sum())
    print("Speakers:", df["speaker"].value_counts().to_dict())