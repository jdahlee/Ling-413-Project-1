from pathlib import Path
import pandas as pd
import random

MIN_DOC_FREQ = 5
META_COLS = ["country_code", "label"]

COUNTRY_DATA_CAP = 8
COUNTRY_DATA_CAP_INCLUDE_ALL_FLAG = -1
ALWAYS_INCLUDE_US = True

def prune_and_load(file: Path) -> pd.DataFrame:
    print(f"Pruning dataset {file}")
    df = pd.read_parquet(file)
    word_cols = df.columns.difference(META_COLS)

    doc_freq = (df[word_cols] > 0).sum(axis=0)
    keep_cols = doc_freq[doc_freq >= MIN_DOC_FREQ].index

    df = df[list(keep_cols) + META_COLS]

    print(f"{file.name}: {df.shape[1] - len(META_COLS)} words kept (of {len(word_cols)})")
    return df

def combine_country_data(data_path: Path, out_path: Path):
    country_files = sorted(data_path.glob("*_word_counts.parquet"))

    if COUNTRY_DATA_CAP != COUNTRY_DATA_CAP_INCLUDE_ALL_FLAG:
        if ALWAYS_INCLUDE_US:
            us_file = next((f for f in country_files if f.stem.startswith("us_")), None)
            other_files = [f for f in country_files if f != us_file]

            selected_others = random.sample(other_files, COUNTRY_DATA_CAP - 1)
            country_files = sorted([us_file] + selected_others)

        else:
            country_files = sorted(random.sample(country_files, COUNTRY_DATA_CAP))

        print(f"Selected countries: {[f.stem for f in country_files]}")

    dfs = []
    for file in country_files:
        print(f"Loading {file.name}")
        df = prune_and_load(file)
        dfs.append(df)

    print("Combining country DataFrames...")
    combined = pd.concat(dfs, axis=0)

    word_cols = combined.columns.difference(["country_code", "label"])
    # Fill in missing vocabulary words as 0s
    print("Filling in NaN values...")
    combined[word_cols] = combined[word_cols].fillna(0).astype(int)

    combined.to_parquet(out_path)
    print(f"Saved combined data to {out_path}")

def main():
    data_path = Path("Country Word Frequency Data")
    out_path = Path("combined_word_counts.parquet")

    if out_path.exists():
        print(f"{out_path} already exists, skipping combine")
        return

    combine_country_data(data_path, out_path)

if __name__ == "__main__":
    main()