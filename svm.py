import os
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def clean_text(text):
    """
    Clean a single text string by converting to lowercase and removing punctuation.

    Args:
        text: The text to clean. If NaN/missing, treated as an empty string.

    Returns:
        str: The cleaned text with punctuation removed and converts to lowercase.
    """
    if pd.isnull(text):
        return ""
    return text.lower().translate(str.maketrans("", "", string.punctuation))

def save_cleaned_column(text_series, save_dir="Data/Processed", filename="svm_cleaned_input_text.txt"):
    """
    Save a list of text strings to a text file, one line per text.

    Args:
        text_series_df: The cleaned text data to save.
        save_dir: Default is 'Data/Processed'.
        filename : Default is 'svm_cleaned_input_text.txt'.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    cleaned_texts = text_series.tolist()
    file_path = os.path.join(save_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        for line in cleaned_texts:
            f.write(line + "\n")
    print(f"Cleaned texts saved to {file_path}")

def load_and_clean(filepath):
    df = pd.read_csv(filepath, encoding="latin1")
    for col in ["title", "text", "tags", "heading", "source"]:
        df[col] = df[col].apply(clean_text)
    df["input_text"] = (
        df["title"] + " " +
        df["text"] + " " +
        df["tags"] + " " +
        df["heading"] + " " +
        df["source"]
    )
    # Save the combined cleaned text
    save_cleaned_column(df["input_text"], filename="svm_cleaned_input_text.txt")
    """
    Load CSV with news article data, clean specified text columns, combine into one input column,
    save combined cleaned text, and drop rows with missing missing labels.

    Args:
        filepath: Path to the input CSV file.

    Returns:
        cleaned DataFrame.
    """
    df = df.dropna(subset=["bias_rating"])
    return df

def split_data(df):
    """
    Split DataFrame into train, validation, and test subsets using stratified sampling on 'bias_rating'.

    Args:
        df: DataFrame with labels.

    Returns:
        train_df, val_df, test_df:  DataFrame.
    """
    train_val, test = train_test_split(
        df, test_size=0.2, random_state=0, stratify=df["bias_rating"]
    )
    train, val = train_test_split(
        train_val, test_size=0.25, random_state=0, stratify=train_val["bias_rating"]
    )
    return train, val, test

def vectorize(train, val, test):
    """
    Fit a vectorizer on the training set and transform train, val, and test sets.

    Args:
        train, val, test: DataFrames with an 'input_text' column.

    Returns:
        X_train, X_val, X_test, vectorizer: the transformed feature matrices and the vectorizer object.
    """
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), lowercase=True)
    X_train = vectorizer.fit_transform(train["input_text"])
    X_val = vectorizer.transform(val["input_text"])
    X_test = vectorizer.transform(test["input_text"])
    return X_train, X_val, X_test, vectorizer


def tune_and_train_svm(X_train, y_train):
    """
    Tune and train an SVM classifier using regularization.
    Saves the best estimator of models to 'Models' directory.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        LinearSVC: best selected model.
    """
    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
    svm = LinearSVC(random_state=0, class_weight="balanced", max_iter=5000, dual="auto")
    final_model = GridSearchCV(svm, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    final_model.fit(X_train, y_train)
    print(f"Best C: {final_model.best_params_['C']}")
    
    # Create models direcxtory
    os.makedirs("Models", exist_ok=True)
    
    # Save the best estimator of model to specified location
    model_path = os.path.join("Models", "best_svm_model.joblib")
    joblib.dump(final_model.best_estimator_, model_path)
    print(f"Model saved to {model_path}")
    
    return final_model.best_estimator_

def save_svm_model(model, X_val, y_val, X_test, y_test, save_dir):
    """
    Evaluate model on validation and test sets, save the classification report and
    confusion matrix plot in the specified directory.

    Args:
        model: Trained SVM model.
        X_val, y_val: Validation feature matrix and labels
        X_test, y_test: Test feature matrix and labels
        save_dir: Output Directory.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    # Validate
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print("Validation Accuracy:", val_acc)
    # Test
    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print("Test Accuracy:", test_acc)
    report = classification_report(y_test, test_preds)
    print("Classification Report:\n", report)
    with open(os.path.join(save_dir, "Traditional Model Classification Report.txt"), "w") as f:
        f.write(report)
    # generate and save Confusion Matrix
    cm = confusion_matrix(y_test, test_preds, labels=model.classes_)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Traditional Model Confusion Matrix (Test Set)")
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "Traditional Model Confusion Matrix.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix plot saved to {plot_path}")