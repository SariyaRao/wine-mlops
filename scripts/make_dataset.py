"""Create data/raw/wine.csv from scikit-learn's built-in Wine dataset.

Run once, then version the file with `dvc add data/raw/wine.csv`.
In a real project this file would come from your own source (database, export...).
"""
from pathlib import Path

from sklearn.datasets import load_wine


def main() -> None:
    df = load_wine(as_frame=True).frame
    df = df.rename(columns={"od280/od315_of_diluted_wines": "od280_od315"})
    out = Path("data/raw/wine.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} with shape {df.shape}")


if __name__ == "__main__":
    main()
