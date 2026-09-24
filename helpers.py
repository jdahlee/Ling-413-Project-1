from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from collections import defaultdict
from pathlib import Path
import pandas as pd
import random
import codecs

# region extract_corpus_documents helpers
def simple_clean_text(text: str) -> str:
    cleaned_text = text.strip().replace("<p>","").replace("\n","").replace("\r","").replace("@","").replace(".","\n")
    return cleaned_text

# endregion

# region collect_all_country_word_data helpers

# Does simple tokenization by stripping, converting to lower case, removing punctuation and returning the word / token array 
def simple_tokenize_text(text: str) -> list:
    cleaned_text = text.strip().lower().replace(",", "").replace(";","").replace(".","").replace("?","").replace("!","").replace("\"","")
    token_array = cleaned_text.split()
    return token_array

def find_word_counts(doc_text: str):
    doc_words = simple_tokenize_text(doc_text)

    freq_dict = defaultdict(int)
    for word in doc_words:
        freq_dict[word] += 1
    
    return freq_dict

def select_country_documents(country_document_cap: int, country_documents: list) -> list:
        print(f"COUNTRY_DOCUMENT_CAP set to {country_document_cap}, randomly selecting {country_document_cap} files")
        random.shuffle(country_documents)
        selected_country_documents = country_documents[:country_document_cap]
        return selected_country_documents

def collect_country_word_data(country_code: str, country_dir: Path, out_file_path: Path, country_code_column_title: str, country_document_cap: int) -> None:
    print(f"Beginning to process word frequency counts for {country_code}...")

    rows = []
    doc_ids = []
    documents_processed = 0

    country_documents = sorted(country_dir.glob("*.txt"))
    if country_document_cap != -1:
        country_documents = select_country_documents(country_document_cap, country_documents)

    for document in country_documents:
        document_text = document.read_text(encoding="utf-8", errors="ignore")
        rows.append(find_word_counts(document_text))
        doc_ids.append(f"{country_code}_{document.stem}")

        documents_processed += 1
        if documents_processed % 100 == 0:
            print(f"Processed {documents_processed} documents for country {country_code}")

    print(f"Creating word frequncy data fram for {country_code}")
    country_word_freq_df = pd.DataFrame(rows, index=doc_ids)
    country_word_freq_df = country_word_freq_df.fillna(0).astype(int)
    country_word_freq_df[country_code_column_title] = country_code

    print("Exporting data frame...")
    country_word_freq_df.to_parquet(out_file_path) # export as parquet file becuase of how sparse + large the dfs are
    print(f"Finished processing {country_code} documents, word frequency data output to {out_file_path}")

# endregion
    
# region prune_and_combine_country_word_data helpers

def select_country_word_frequency_files(country_files: list, country_data_cap: int, always_include_us: bool) -> list:
    if always_include_us:
        us_file = next((f for f in country_files if f.stem.startswith("us_")), None)
        other_files = [f for f in country_files if f != us_file]

        selected_others = random.sample(other_files, country_data_cap - 1)
        selected_country_files = sorted([us_file] + selected_others)

    else:
        selected_country_files = sorted(random.sample(country_files, country_data_cap))

    print(f"Selected countries: {[f.stem for f in selected_country_files]}")
    return selected_country_files

def prune_and_load_data_frame(file: Path, meta_columns: list, min_doc_freq: int) -> pd.DataFrame:
    print(f"Pruning dataset {file}")
    df = pd.read_parquet(file)
    word_cols = df.columns.difference(meta_columns)

    doc_freq = (df[word_cols] > 0).sum(axis=0)
    keep_cols = doc_freq[doc_freq >= min_doc_freq].index

    df = df[list(keep_cols) + meta_columns]

    print(f"{file.name}: {df.shape[1] - len(meta_columns)} words kept (of {len(word_cols)})")
    return df

# endregion

# region train_and_evaluate_models helpers

def apply_tf_idf_transformation(x_train: pd.DataFrame, x_test: pd.DataFrame):
    print("Applying TF-IDF transformation to data...")
    tf_idf_transformer = TfidfTransformer()

    x_train_df_sparse = x_train.astype(pd.SparseDtype("float64", 0))
    x_train_sparse_matrix = x_train_df_sparse.sparse.to_coo().tocsr()
    tf_idf_transformed_x_train = tf_idf_transformer.fit_transform(x_train_sparse_matrix)

    x_test_df_sparse = x_test.astype(pd.SparseDtype("float64", 0))
    x_test_sparse_matrix = x_test_df_sparse.sparse.to_coo().tocsr()
    tf_idf_transformed_x_test = tf_idf_transformer.transform(x_test_sparse_matrix)

    return tf_idf_transformed_x_train, tf_idf_transformed_x_test

def evaluate_and_output_results(y_test: list, y_pred: list, output_file_name) -> None:
    report = classification_report(y_test, y_pred)
    print (f"Writing resuts to {output_file_name}")
    with codecs.open(output_file_name, "w", encoding = "utf-8") as f:
        f.write(report)

# endregion