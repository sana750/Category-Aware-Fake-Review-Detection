# Converted from Category_Aware_Fake_Review_Detection_in_English (1).ipynb

# %% [cell 1]

# Data handling
import pandas as pd
import numpy as np

# Machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Save trained model
import joblib

from pathlib import Path

# Colab's display() helper is not available in a normal Python script.
try:
    display
except NameError:
    display = lambda value: print(value)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed" / "processed"
COLAB_DATA_DIR = Path("/content/drive/MyDrive/Category-Aware Fake Review Detection/data/processed/processed")
if not (DATA_DIR / "en_train.csv").exists() and (COLAB_DATA_DIR / "en_train.csv").exists():
    DATA_DIR = COLAB_DATA_DIR

print("Libraries imported successfully!")

# %% [cell 2]

# Check dataset shapes
en_train = pd.read_csv(DATA_DIR / "en_train.csv")
en_val = pd.read_csv(DATA_DIR / "en_val.csv")
en_test = pd.read_csv(DATA_DIR / "en_test.csv")
print("Training data:", en_train.shape)
print("Validation data:", en_val.shape)
print("Testing data:", en_test.shape)

# Display first 5 rows
display(en_train.head())

# Check column names
print("\nColumns:", en_train.columns.tolist())

# %% [cell 3]

print("Training label distribution:")
print(en_train['label'].value_counts())

print("\nTraining label percentages:")
print(en_train['label'].value_counts(normalize=True) * 100)

print("\nUnique labels:")
print(en_train['label'].unique())

# %% [cell 4]

# Extract review text
X_train = en_train['review_text']
X_val = en_val['review_text']
X_test = en_test['review_text']

# Extract labels
y_train = en_train['label']
y_val = en_val['label']
y_test = en_test['label']

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# %% [cell 5]

# fake/genuine label classification:

# %% [cell 6]

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    lowercase=True,
    strip_accents='unicode',
    ngram_range=(1, 2),
    max_features=100000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

# Fit only on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform validation and testing data
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF conversion completed!")

print("Training matrix:", X_train_tfidf.shape)
print("Validation matrix:", X_val_tfidf.shape)
print("Testing matrix:", X_test_tfidf.shape)

# %% [cell 7]

# Initialize Logistic Regression
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train model
lr_model.fit(X_train_tfidf, y_train)

print("Logistic Regression training completed!")

# %% [cell 8]

# IMPORTANT:
# Change this if your Part 1 label mapping is different.
# Example: 1 = Fake, 0 = Genuine

FAKE_LABEL = 1

# Check that the fake label exists
assert FAKE_LABEL in lr_model.classes_, "Fake label not found!"

# Get fake-review probabilities
fake_class_index = list(lr_model.classes_).index(FAKE_LABEL)

val_probabilities = lr_model.predict_proba(X_val_tfidf)[:, fake_class_index]

# Classification threshold
threshold = 0.5

# Generate predictions
val_predictions = np.where(
    val_probabilities >= threshold,
    FAKE_LABEL,
    next(label for label in lr_model.classes_ if label != FAKE_LABEL)
)

# Calculate evaluation metrics
print("VALIDATION RESULTS")
print("-------------------")

print("Accuracy:", accuracy_score(y_val, val_predictions))

print("Precision:", precision_score(
    y_val, val_predictions, pos_label=FAKE_LABEL
))

print("Recall:", recall_score(
    y_val, val_predictions, pos_label=FAKE_LABEL
))

print("F1 Score:", f1_score(
    y_val, val_predictions, pos_label=FAKE_LABEL
))

print("ROC-AUC:", roc_auc_score(
    (y_val == FAKE_LABEL).astype(int),
    val_probabilities
))

# %% [cell 9]

genuine_label = next(label for label in lr_model.classes_
    if label != FAKE_LABEL)

print(classification_report(
    y_val,
    val_predictions,
    labels=[genuine_label, FAKE_LABEL],
    target_names=['Genuine', 'Fake'],
    zero_division=0
))

# %% [cell 10]

cm = confusion_matrix(
    y_val,
    val_predictions,
    labels=[genuine_label, FAKE_LABEL]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Genuine', 'Fake'],
    yticklabels=['Genuine', 'Fake']
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("TF-IDF + Logistic Regression - Validation Confusion Matrix")

plt.show()

# %% [cell 11]

# Predict probabilities on test data
test_probabilities = lr_model.predict_proba(X_test_tfidf)[:, fake_class_index]

# Predict labels
test_predictions = np.where(
    test_probabilities >= threshold,
    FAKE_LABEL,
    genuine_label
)

# Calculate metrics
print("FINAL TEST RESULTS")
print("-------------------")

print("Accuracy:", accuracy_score(y_test, test_predictions))

print("Precision:", precision_score(
    y_test, test_predictions, pos_label=FAKE_LABEL
))

print("Recall:", recall_score(
    y_test, test_predictions, pos_label=FAKE_LABEL
))

print("F1 Score:", f1_score(
    y_test, test_predictions, pos_label=FAKE_LABEL
))

print("ROC-AUC:", roc_auc_score(
    (y_test == FAKE_LABEL).astype(int),
    test_probabilities
))

print("\nClassification Report:")

print(classification_report(
    y_test,
    test_predictions,
    labels=[genuine_label, FAKE_LABEL],
    target_names=['Genuine', 'Fake'],
    zero_division=0
))

# %% [cell 12]

# Save TF-IDF vectorizer
joblib.dump(tfidf, PROJECT_DIR / 'english_tfidf_vectorizer.pkl')

# Save Logistic Regression model
joblib.dump(lr_model, PROJECT_DIR / 'english_logistic_regression.pkl')

print("Models saved successfully!")

# %% [cell 13]

fake_review_df = pd.DataFrame({
    'review_text': en_test['review_text'].values,
    'category': en_test['category'].values,
    'original_label': y_test.values,
    'predicted_label': test_predictions
})

# Convert labels into readable names
fake_review_df['original_label'] = fake_review_df['original_label'].map({
    0: 'Genuine',
    1: 'Fake'
})

fake_review_df['predicted_label'] = fake_review_df['predicted_label'].map({
    0: 'Genuine',
    1: 'Fake'
})

display(fake_review_df.head(20))

# %% [cell 14]

# Product Category Prediction

# %% [cell 15]

print("Number of categories:", en_train['category'].nunique())

print("\nCategory distribution:")
print(en_train['category'].value_counts())

# %% [cell 16]

# Category labels
y_train_cat = en_train['category']
y_val_cat = en_val['category']
y_test_cat = en_test['category']

print("Training categories:", y_train_cat.nunique())
print("Validation categories:", y_val_cat.nunique())
print("Testing categories:", y_test_cat.nunique())

# %% [cell 17]

# Initialize category classifier
category_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

# Train using review text features
category_model.fit(
    X_train_tfidf,
    y_train_cat
)

print("Category model training completed!")

# %% [cell 18]

from sklearn.metrics import accuracy_score, f1_score

# Validation predictions
val_cat_predictions = category_model.predict(X_val_tfidf)

# Evaluation
print("CATEGORY MODEL - VALIDATION RESULTS")
print("-----------------------------------")

print("Accuracy:",
      accuracy_score(y_val_cat, val_cat_predictions))

print("Macro F1:",
      f1_score(
          y_val_cat,
          val_cat_predictions,
          average='macro'
      ))

print("\nClassification Report:")

print(classification_report(
    y_val_cat,
    val_cat_predictions,
    zero_division=0
))

# %% [cell 19]

# Create a DataFrame with actual and predicted categories
category_results = pd.DataFrame({
    'review_text': en_val['review_text'].values,
    'original_category': en_val['category'].values,
    'predicted_category': val_cat_predictions
})

# Check whether prediction is correct
category_results['correct_prediction'] = (category_results['original_category']==category_results['predicted_category'])

# Display results
display(category_results.head(20))

# %% [cell 20]

category_results.to_csv(PROJECT_DIR / "english_category_predictions.csv", index=False)

print("Category prediction results saved!")

# %% [cell 21]

from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(12, 10))

ConfusionMatrixDisplay.from_predictions(
    y_val_cat,
    val_cat_predictions,
    xticks_rotation=90,
    cmap='Blues',
    ax=ax
)

plt.title("Product Category Prediction - Validation")
plt.tight_layout()
plt.show()

# %% [cell 22]

# Category-Aware Fake Review Detection.

# %% [cell 23]

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import clone

# Five-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Generate out-of-fold category predictions
train_cat_predictions = cross_val_predict(
    category_model,
    X_train_tfidf,
    y_train_cat,
    cv=cv,
    method='predict',
    n_jobs=-1
)

print("Training category predictions completed!")

# Check some predictions
category_check = pd.DataFrame({
    'review_text': en_train['review_text'].values,
    'original_category': y_train_cat.values,
    'predicted_category': train_cat_predictions
})

category_check['correct_prediction'] = (
    category_check['original_category'] ==
    category_check['predicted_category']
)

display(category_check.head(10))

print("\nOOF Category Accuracy:",
      accuracy_score(y_train_cat, train_cat_predictions))

# %% [cell 24]

# Predict categories for validation data
val_cat_predictions = category_model.predict(X_val_tfidf)

# Predict categories for test data
test_cat_predictions = category_model.predict(X_test_tfidf)

print("Validation category predictions completed!")
print("Test category predictions completed!")

# %% [cell 25]

category_predictions_df = pd.DataFrame({
    'review_text': en_val['review_text'].values,
    'original_category': en_val['category'].values,
    'predicted_category': val_cat_predictions
})

display(category_predictions_df.head(10))

# %% [cell 26]

from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

# Initialize encoder
category_encoder = OneHotEncoder(
    handle_unknown='ignore'
)

# Convert training categories into a DataFrame
train_cat_df = pd.DataFrame({
    'category': train_cat_predictions
})

# Fit encoder on training category predictions
X_train_cat = category_encoder.fit_transform(train_cat_df)

# Transform validation categories
val_cat_df = pd.DataFrame({
    'category': val_cat_predictions
})

X_val_cat = category_encoder.transform(val_cat_df)

# Transform test categories
test_cat_df = pd.DataFrame({
    'category': test_cat_predictions
})

X_test_cat = category_encoder.transform(test_cat_df)

print("Category encoding completed!")

print("Training category features:", X_train_cat.shape)
print("Validation category features:", X_val_cat.shape)
print("Testing category features:", X_test_cat.shape)

# %% [cell 27]

# Combine text and category features
X_train_combined = hstack([
    X_train_tfidf,
    X_train_cat
]).tocsr()

X_val_combined = hstack([
    X_val_tfidf,
    X_val_cat
]).tocsr()

X_test_combined = hstack([
    X_test_tfidf,
    X_test_cat
]).tocsr()

print("Combined feature matrices created!")

print("Training:", X_train_combined.shape)
print("Validation:", X_val_combined.shape)
print("Testing:", X_test_combined.shape)

# %% [cell 28]

# Initialize category-aware classifier
category_aware_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train the model
category_aware_model.fit(
    X_train_combined,
    y_train
)

print("Category-aware fake review model trained successfully!")

# %% [cell 29]

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Identify fake and genuine labels
FAKE_LABEL = 1
GENUINE_LABEL = 0

# Get probability column for fake reviews
fake_index_aware = list(category_aware_model.classes_).index(FAKE_LABEL)

# Predict fake probabilities
val_aware_probabilities = category_aware_model.predict_proba(
    X_val_combined
)[:, fake_index_aware]

# Predict labels using threshold 0.5
val_aware_predictions = np.where(
    val_aware_probabilities >= 0.5,
    FAKE_LABEL,
    GENUINE_LABEL
)

# Evaluate
print("CATEGORY-AWARE MODEL - VALIDATION RESULTS")
print("------------------------------------------")

print("Accuracy:",
      accuracy_score(y_val, val_aware_predictions))

print("Precision:",
      precision_score(y_val, val_aware_predictions, pos_label=FAKE_LABEL))

print("Recall:",
      recall_score(y_val, val_aware_predictions, pos_label=FAKE_LABEL))

print("F1 Score:",
      f1_score(y_val, val_aware_predictions, pos_label=FAKE_LABEL))

print("ROC-AUC:",
      roc_auc_score(
          (y_val == FAKE_LABEL).astype(int),
          val_aware_probabilities
      ))

print("\nClassification Report:")

print(classification_report(
    y_val,
    val_aware_predictions,
    labels=[GENUINE_LABEL, FAKE_LABEL],
    target_names=['Genuine', 'Fake'],
    zero_division=0
))

# %% [cell 30]

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Identify fake and genuine labels
FAKE_LABEL = 1
GENUINE_LABEL = 0

# Get probability column for fake reviews
fake_index_aware = list(category_aware_model.classes_).index(FAKE_LABEL)

# Predict fake probabilities
val_aware_probabilities = category_aware_model.predict_proba(
    X_val_combined
)[:, fake_index_aware]

# Predict labels using threshold 0.5
val_aware_predictions = np.where(
    val_aware_probabilities >= 0.5,
    FAKE_LABEL,
    GENUINE_LABEL
)

# Evaluate
print("CATEGORY-AWARE MODEL - VALIDATION RESULTS")
print("------------------------------------------")

print("Accuracy:",
      accuracy_score(y_val, val_aware_predictions))

print("Precision:",
      precision_score(y_val, val_aware_predictions, pos_label=FAKE_LABEL))

print("Recall:",
      recall_score(y_val, val_aware_predictions, pos_label=FAKE_LABEL))

print("F1 Score:",
      f1_score(y_val, val_aware_predictions, pos_label=FAKE_LABEL))

print("ROC-AUC:",
      roc_auc_score(
          (y_val == FAKE_LABEL).astype(int),
          val_aware_probabilities
      ))

print("\nClassification Report:")

print(classification_report(
    y_val,
    val_aware_predictions,
    labels=[GENUINE_LABEL, FAKE_LABEL],
    target_names=['Genuine', 'Fake'],
    zero_division=0
))

# %% [cell 31]

# Predict fake probabilities
test_aware_probabilities = category_aware_model.predict_proba(
    X_test_combined
)[:, fake_index_aware]

# Predict fake/genuine labels
test_aware_predictions = np.where(
    test_aware_probabilities >= 0.5,
    FAKE_LABEL,
    GENUINE_LABEL
)

# Evaluate
print("CATEGORY-AWARE MODEL - FINAL TEST RESULTS")
print("-----------------------------------------")

print("Accuracy:",
      accuracy_score(y_test, test_aware_predictions))

print("Precision:",
      precision_score(y_test, test_aware_predictions, pos_label=FAKE_LABEL))

print("Recall:",
      recall_score(y_test, test_aware_predictions, pos_label=FAKE_LABEL))

print("F1 Score:",
      f1_score(y_test, test_aware_predictions, pos_label=FAKE_LABEL))

print("ROC-AUC:",
      roc_auc_score(
          (y_test == FAKE_LABEL).astype(int),
          test_aware_probabilities
      ))

print("\nClassification Report:")

print(classification_report(
    y_test,
    test_aware_predictions,
    labels=[GENUINE_LABEL, FAKE_LABEL],
    target_names=['Genuine', 'Fake'],
    zero_division=0
))

# %% [cell 32]

# Find the column index corresponding to the Fake label
FAKE_LABEL = 1

fake_index_baseline = list(lr_model.classes_).index(FAKE_LABEL)

print("Model classes:", lr_model.classes_)
print("Fake label index:", fake_index_baseline)

# %% [cell 33]

# Baseline Model 1 test probabilities
baseline_test_probabilities = lr_model.predict_proba(
    X_test_tfidf
)[:, fake_index_baseline]

baseline_test_predictions = np.where(
    baseline_test_probabilities >= 0.5,
    FAKE_LABEL,
    GENUINE_LABEL
)

# Create final comparison
test_comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],

    'Model 1 - Text Only': [
        accuracy_score(y_test, baseline_test_predictions),
        precision_score(y_test, baseline_test_predictions, pos_label=FAKE_LABEL),
        recall_score(y_test, baseline_test_predictions, pos_label=FAKE_LABEL),
        f1_score(y_test, baseline_test_predictions, pos_label=FAKE_LABEL),
        roc_auc_score((y_test == FAKE_LABEL).astype(int),
                      baseline_test_probabilities)
    ],

    'Model 3 - Text + Category': [
        accuracy_score(y_test, test_aware_predictions),
        precision_score(y_test, test_aware_predictions, pos_label=FAKE_LABEL),
        recall_score(y_test, test_aware_predictions, pos_label=FAKE_LABEL),
        f1_score(y_test, test_aware_predictions, pos_label=FAKE_LABEL),
        roc_auc_score((y_test == FAKE_LABEL).astype(int),
                      test_aware_probabilities)
    ]
})

display(test_comparison.round(4))

# %% [cell 34]

baseline_f1 = f1_score(
    y_test, baseline_test_predictions, pos_label=FAKE_LABEL
)

aware_f1 = f1_score(
    y_test, test_aware_predictions, pos_label=FAKE_LABEL
)

improvement = (aware_f1 - baseline_f1) * 100

print(f"Baseline F1 Score: {baseline_f1:.4f}")
print(f"Category-Aware F1 Score: {aware_f1:.4f}")
print(f"F1 Improvement: {improvement:+.2f} percentage points")

# %% [cell 35]

cm = confusion_matrix(
    y_test,
    test_aware_predictions,
    labels=[GENUINE_LABEL, FAKE_LABEL]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Genuine', 'Fake'],
    yticklabels=['Genuine', 'Fake']
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Category-Aware Fake Review Detection")
plt.show()

# %% [cell 36]

category_aware_results = pd.DataFrame({
    'review_text': en_test['review_text'].values,
    'original_category': en_test['category'].values,
    'predicted_category': test_cat_predictions,
    'original_label': y_test.values,
    'predicted_label': test_aware_predictions
})

category_aware_results['original_label'] = (
    category_aware_results['original_label']
    .map({0: 'Genuine', 1: 'Fake'})
)

category_aware_results['predicted_label'] = (
    category_aware_results['predicted_label']
    .map({0: 'Genuine', 1: 'Fake'})
)

category_aware_results['correct_prediction'] = (
    category_aware_results['original_label'] ==
    category_aware_results['predicted_label']
)

display(category_aware_results.head(20))
