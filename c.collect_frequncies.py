from collections import defaultdict
from pathlib import Path
import pandas as pd
import os
import random

COUNTRY_DOCUMENT_NO_CAP_FLAG = -1
COUNTRY_DOCUMENT_CAP = 2500 # Set to COUNTRY_DOCUMENT_NO_CAP_FLAG to remove cap

def find_word_counts(doc_text: str):
    # Basic word cleaning + normalization
    doc_text = doc_text.strip().lower().replace(",", "").replace(";","").replace(".","").replace("?","").replace("!","").replace("\"","")
    doc_words = doc_text.split()

    freq_dict = defaultdict(int)
    for word in doc_words:
        freq_dict[word] += 1
    
    return freq_dict

def process_country(country_code: str, country_dir: Path, out_file_path: Path):
    print(f"Beginning to process {country_code}")

    rows = []
    doc_ids = []
    documents_processed = 0

    files = sorted(country_dir.glob("*.txt"))
    if COUNTRY_DOCUMENT_CAP != COUNTRY_DOCUMENT_NO_CAP_FLAG:
        print(f"COUNTRY_DOCUMENT_CAP set to {COUNTRY_DOCUMENT_CAP}, randomly selecting {COUNTRY_DOCUMENT_CAP} files")
        random.shuffle(files)
        files = files[:COUNTRY_DOCUMENT_CAP]

    for file in files:
        text = file.read_text(encoding="utf-8", errors="ignore")
        rows.append(find_word_counts(text))
        doc_ids.append(f"{country_code}_{file.stem}")

        documents_processed += 1
        if documents_processed % 100 == 0:
            print(f"Processed {documents_processed} documents for country {country_code}")

    df = pd.DataFrame(rows, index=doc_ids)
    df = df.fillna(0).astype(int)
    df["country_code"] = country_code
    df["label"] = "US" if country_code.lower() == "us" else "Non-US"
    print("Exporting data frame..")
    df.to_parquet(out_file_path) # export as parquet file becuase of how sparse + large the dfs are
    print(f"Finished processing {country_code} documents, word frequency data output to {out_file_path}")

def main():
    # Make directory to output frequency data files
    des_folder = "Country Word Frequency Data"
    os.makedirs(des_folder, exist_ok=True)

    corpus_dir = Path("Country Corpus")
    for country_dir in corpus_dir.iterdir():
        if not country_dir.is_dir():
            continue
        country_code = country_dir.name

        out_file_name = f"{country_code}_word_counts.parquet"
        out_file_path = Path(des_folder, out_file_name)
        if out_file_path.exists():
            print(f"Skipping {country_code}, already processed")
            continue
        
        process_country(country_code, country_dir, out_file_path)

if __name__ == "__main__":
    main()
