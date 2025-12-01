from pathlib import Path
from typing import Optional, List
import pandas as pd

def _parse_scores(s: str) -> Optional[List[int]]:
    if s is None or not s.strip():
        return None
    try:
        return [int(x) for x in s.split(",")]
    except ValueError:
        return None

def load_ccpe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    rows = []
    conv_id = 0
    turn_id = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if turn_id > 0:
                    conv_id += 1
                    turn_id = 0
                continue

            parts = line.split("\t")
            while len(parts) < 4:
                parts.append("")
            speaker, text, entity_tag, scores_str = parts[:4]

            turn_id += 1
            scores_raw = scores_str.strip() if scores_str else ""
            scores = _parse_scores(scores_raw)

            rows.append({
                "conv_id": conv_id,
                "turn_id": turn_id,
                "speaker": speaker,
                "text": text,
                "entity_tag": entity_tag if entity_tag else None,
                "scores_raw": scores_raw if scores_raw else None,
                "scores": scores,
                "is_overall": (speaker == "USER" and text == "OVERALL"),
            })

            # Optionally increment conv_id on USER OVERALL
            if speaker == "USER" and text == "OVERALL":
                conv_id += 1
                turn_id = 0

    df = pd.DataFrame(rows)
    return df

def quick_stats(df):
    print("Num rows:", len(df))
    print("Num conversations:", df["conv_id"].nunique())
    print("Num OVERALL lines:", df["is_overall"].sum())
    print("Speakers:", df["speaker"].value_counts().to_dict())
