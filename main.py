from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from enum import Flag, auto
from pathlib import Path
import pandas as pd
import helpers
import zipfile
import codecs
import os

# region Program Params

# Corpus Extraction Params
CORPUS_DOCUMENTS_FOLDER = "Country Corpus"
GLOWBE_DATA_FOLDER_PATH = "GlowBe_Data"

# Word Frequncy Extraction Params
COUNTRY_DOCUMENT_NO_CAP_FLAG = -1
COUNTRY_DOCUMENT_CAP = 2500 # Set to COUNTRY_DOCUMENT_NO_CAP_FLAG to remove cap
COUNTRY_CODE_COLUMN_TITLE = "Country Code"
COUNTRY_WORD_FREQUENCY_DATA_FOLDER = "Country Word Frequency Data"

# Prune And Combine Country Word Data Params
MIN_DOC_FREQ = 5 # How many docs within a country's corpus a word must appear in for us to not prune it
META_COLS = [COUNTRY_CODE_COLUMN_TITLE]
COUNTRY_DATA_CAP_INCLUDE_ALL_FLAG = -1
# How many countries we want to include in our combined data set, set to COUNTRY_DATA_CAP_INCLUDE_ALL_FLAG to include all countries
COUNTRY_DATA_CAP = 8 
ALWAYS_INCLUDE_US = True # Always include US in combined data set
COMBINED_WORD_DATA_FILE = "combined_word_counts.parquet"

# endregion

# region Program Steps

def extract_corpus_documents() -> None:
    print(f"Beginning corpus document extraction, outputing to {CORPUS_DOCUMENTS_FOLDER}")
    os.makedirs(CORPUS_DOCUMENTS_FOLDER, exist_ok=True)

    print(f"Iterating through zip files in {GLOWBE_DATA_FOLDER_PATH}...")
    for corpus_file in os.listdir(GLOWBE_DATA_FOLDER_PATH):
        if corpus_file.endswith(".zip"):
            country = corpus_file.split("_")[1]
            print(f"Processing documents for {country}...")

            with zipfile.ZipFile(os.path.join(GLOWBE_DATA_FOLDER_PATH, corpus_file), "r") as zf:
                for name in zf.namelist():

                    if name.endswith(".txt"):
                        path = zipfile.Path(zf, at=name)
                        with path.open("r", encoding = "utf-8") as f:
                            print(f"Processing corpus documents in {name}")
                            corpus_documents_processed = 0
                            for corpus_document in f:

                                if len(corpus_document) > 10: # skip over blank lines
                                    text_id = corpus_document.split()[0].replace("#","")
                                    text = corpus_document[2:].replace(text_id, "")
                                    
                                    cleaned_text = helpers.simple_clean_text(text)
                                    write_name = os.path.join(CORPUS_DOCUMENTS_FOLDER, str(text_id)+"." + country + ".txt") # e.g. 123.us.txt

                                    with codecs.open(write_name, "w", encoding = "utf-8") as fw:
                                            if len(cleaned_text) > 5:
                                                fw.write(cleaned_text)
                                                corpus_documents_processed += 1
                                                if corpus_documents_processed % 1000 == 0: print(f"Procesed {corpus_documents_processed} for {name}")
    print("Finished corpus document extraction")

def organize_corpus_documents() -> None:
    print("Beginning corpus document organization by country")
    
    documents_processed = 0
    for file in os.listdir(CORPUS_DOCUMENTS_FOLDER):
        if file.endswith(".txt"):
            country = file.split(".")[1]
            os.makedirs(os.path.join(CORPUS_DOCUMENTS_FOLDER, country), exist_ok=True)
            os.rename(os.path.join(CORPUS_DOCUMENTS_FOLDER, file), os.path.join(CORPUS_DOCUMENTS_FOLDER, country, file))

            documents_processed += 1
            if (documents_processed % 10000 == 0):
                print(f"documents_processed documents organized")

def collect_all_country_word_data() -> None:
    os.makedirs(COUNTRY_WORD_FREQUENCY_DATA_FOLDER, exist_ok=True)

    corpus_dir = Path(CORPUS_DOCUMENTS_FOLDER)
    for country_dir in corpus_dir.iterdir():
        if not country_dir.is_dir():
            continue
        country_code = country_dir.name

        out_file_name = f"{country_code}_word_counts.parquet"
        out_file_path = Path(COUNTRY_WORD_FREQUENCY_DATA_FOLDER, out_file_name)
        if out_file_path.exists():
            print(f"Skipping {country_code}, already processed")
            continue
        
        helpers.collect_country_word_data(country_code, country_dir, out_file_path, 
                                          COUNTRY_CODE_COLUMN_TITLE, COUNTRY_DOCUMENT_CAP)

def prune_and_combine_country_word_data() -> None:
    out_path = Path(COMBINED_WORD_DATA_FILE)
    if out_path.exists():
        print(f"{out_path} already exists, skipping combine")
        return

    country_word_frequency_data_path = Path(COUNTRY_WORD_FREQUENCY_DATA_FOLDER)

    country_files = sorted(country_word_frequency_data_path.glob("*_word_counts.parquet"))

    if COUNTRY_DATA_CAP != COUNTRY_DATA_CAP_INCLUDE_ALL_FLAG:
        country_files = helpers.select_country_word_frequency_files(country_files, COUNTRY_DATA_CAP, ALWAYS_INCLUDE_US)

    dfs = []
    for file in country_files:
        print(f"Loading {file.name}")
        df = helpers.prune_and_load_data_frame(file, META_COLS, MIN_DOC_FREQ)
        dfs.append(df)

    print("Combining country DataFrames...")
    combined = pd.concat(dfs, axis=0)

    word_cols = combined.columns.difference(META_COLS)
    # Fill in missing vocabulary words as 0s
    print("Filling in NaN values...")
    combined[word_cols] = combined[word_cols].fillna(0).astype(int)

    combined.to_parquet(out_path)
    print(f"Saved combined data to {out_path}")

def train_and_evaluate_models() -> None:
    print(f"Reading in combined data from {COMBINED_WORD_DATA_FILE}")
    combined_word_data_df = pd.read_parquet(COMBINED_WORD_DATA_FILE)

    data_x = combined_word_data_df.drop(META_COLS, axis = 1).reset_index(drop=True)
    data_y = combined_word_data_df.loc[:,COUNTRY_CODE_COLUMN_TITLE].reset_index(drop=True)   
    test_size = 0.10
    print(f"Splitting training and testing data with test_size {test_size}")
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_size, shuffle=True, stratify=data_y)

    tf_idf_transformed_x_train, tf_idf_transformed_x_test = helpers.apply_tf_idf_transformation(x_train, x_test)

    # Train logistic regression classifier
    print("Training classifier...")
    logistic_regression_classifier = LogisticRegression(max_iter = 100000)
    logistic_regression_classifier.fit(tf_idf_transformed_x_train, y_train)
    print("Using model to predict test data...")
    logistic_regression_y_preds = logistic_regression_classifier.predict(tf_idf_transformed_x_test)

    logistic_regression_classifier_results_file = "logistic_regression_classification_results.txt"
    helpers.evaluate_and_output_results(y_test, logistic_regression_y_preds, logistic_regression_classifier_results_file)

    # Train dummy model
    print("Training dummy model...")
    # Using a uniform strategy for the dummy classifier becuase we have an even number of documents between all countries
    dummy_classifier = DummyClassifier(strategy="uniform")
    dummy_classifier.fit(tf_idf_transformed_x_train, y_train)
    print("Using dummy classifier to predict the test data...")
    dummy_y_preds = dummy_classifier.predict(tf_idf_transformed_x_test)

    dummy_classifier_results_file = "dummy_classification_results.txt"
    helpers.evaluate_and_output_results(y_test, dummy_y_preds, dummy_classifier_results_file)

# endregion

# Program Steps Param
class Step(Flag):
    EXTRACT_CORPUS = auto()     
    ORGANIZE_CORPUS = auto()     
    COLLECT_WORD_DATA = auto()   
    PRUNE_AND_COMBINE = auto()   
    TRAIN_AND_EVALUATE = auto()  
    ALL = EXTRACT_CORPUS | ORGANIZE_CORPUS | COLLECT_WORD_DATA | PRUNE_AND_COMBINE | TRAIN_AND_EVALUATE

# Adjust this to the steps that you want to run
STEPS_TO_RUN = Step.COLLECT_WORD_DATA | Step.PRUNE_AND_COMBINE | Step.TRAIN_AND_EVALUATE

def main():
    if Step.EXTRACT_CORPUS in STEPS_TO_RUN:
        extract_corpus_documents()
    if Step.ORGANIZE_CORPUS in STEPS_TO_RUN:
        organize_corpus_documents()
    if Step.COLLECT_WORD_DATA in STEPS_TO_RUN:
        collect_all_country_word_data()
    if Step.PRUNE_AND_COMBINE in STEPS_TO_RUN:
        prune_and_combine_country_word_data()
    if Step.TRAIN_AND_EVALUATE in STEPS_TO_RUN:
        train_and_evaluate_models()

if __name__ == "__main__":
    main()