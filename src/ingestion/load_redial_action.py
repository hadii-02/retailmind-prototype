import pandas as pd
from pathlib import Path

def load_redial_action(path: str | Path) -> pd.DataFrame:
    """
    Parse ReDial-Action.txt:
    Format:
    SPEAKER \t TEXT \t DIALOG_ACT
    """

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
            while len(parts) < 3:
                parts.append("")

            speaker, text, act = parts[:3]

            turn_id += 1

            rows.append({
                "conv_id": conv_id,
                "turn_id": turn_id,
                "speaker": speaker,
                "text": text,
                "dialog_act": act if act else None,
                "scores_raw": None,
                "scores": None,
                "is_overall": False,
            })

    return pd.DataFrame(rows)


def quick_stats(df):
    print("Rows:", len(df))
    print("Conversations:", df["conv_id"].nunique())
    print("Speakers:", df["speaker"].value_counts().to_dict())
