"""Stage 1: split raw data into train/test sets."""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    df = pd.read_csv("data/raw/wine.csv")

    train, test = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df["target"],
    )

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print(f"train={train.shape} test={test.shape}")


if __name__ == "__main__":
    main()
