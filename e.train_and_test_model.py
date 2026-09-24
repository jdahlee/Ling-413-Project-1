import pandas as pd
import codecs
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier

combined_word_counts_file = "combined_word_counts.parquet"

print(f"Reading in combined data from {combined_word_counts_file}")
data_df = pd.read_parquet(combined_word_counts_file)

data_x = data_df.drop(["country_code", "label"], axis = 1).reset_index(drop=True)
data_y = data_df.loc[:,"country_code"].reset_index(drop=True)   
test_size = 0.10
print(f"Splitting training and testing data with test_size {test_size}")
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_size, shuffle=True, stratify=data_y)

# Convert to sparse matrix for TfidfTransformer
print("Applying TF-IDF transformation to data...")
tf_idf_transformer = TfidfTransformer()

x_train_df_sparse = x_train.astype(pd.SparseDtype("float64", 0))
x_train_sparse_matrix = x_train_df_sparse.sparse.to_coo().tocsr()
tf_idf_transformed_x_train = tf_idf_transformer.fit_transform(x_train_sparse_matrix)

x_test_df_sparse = x_test.astype(pd.SparseDtype("float64", 0))
x_test_sparse_matrix = x_test_df_sparse.sparse.to_coo().tocsr()
tf_idf_transformed_x_test = tf_idf_transformer.transform(x_test_sparse_matrix)

# Train model
print("Training model...")
cls = LogisticRegression(max_iter = 100000)
cls.fit(tf_idf_transformed_x_train, y_train)

print("Using model to predict test data...")
y_pred = cls.predict(tf_idf_transformed_x_test)

report = classification_report(y_test, y_pred)
results_file = "classification_results.txt"
print (f"Writing resuts to {results_file}")
with codecs.open(results_file, "w", encoding = "utf-8") as f:
    f.write(report)

# Train dummy classifier
print("Training dummy classifier...")
dummy_classifier = DummyClassifier(strategy="uniform")
dummy_classifier.fit(tf_idf_transformed_x_train, y_train)

print("Using dummy classifier to predict the test data...")
y_pred = dummy_classifier.predict(tf_idf_transformed_x_test)

report = classification_report(y_test, y_pred)
dummy_results_file = "dummy_" + results_file
print (f"Writing dummy resuts to {results_file}")
with codecs.open(dummy_results_file, "w", encoding = "utf-8") as f:
    f.write(report)