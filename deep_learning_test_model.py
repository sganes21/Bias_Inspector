import os
import json
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_unlabeled_texts(filepath):
    """
    Load unlabeled test article user messages from a .json file.

    Args:
        filepath: Path to the file containing test data without labels.

    Returns:
        list of user message strings representing the news articles to classify.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    texts = []
    for item in data:
        user_message = next(m['content'] for m in item['messages'] if m['role'] == 'user')
        texts.append(user_message)
    print(f"Loaded {len(texts)} unlabeled test articles.")
    return texts

def load_ground_truth_labels(filepath):
    """
    Load ground truth  labels from a .json file.

    Args:
        filepath: Path to the .json file with labeled data.

    Returns:
        A list of ground truth bias labels ('left', 'center', 'right', or None for missing).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    true_labels = []
    for item in data:
        assistant_message = next((m['content'] for m in item['messages'] if m['role'] == 'assistant'), None)
        true_labels.append(assistant_message.strip().lower() if assistant_message else None)
    print(f"Loaded {len(true_labels)} ground truth labels.")
    return true_labels

def get_prediction_for_text(client, model_name, article_text):
    """
    Input news article to an OpenAI model to predict its bias label.

    Args:
        client: The authenticated OpenAI API client.
        model_name: The ID of the fine-tuned model to use for predictions.
        The news article data.

    Returns:
       The predicted bias label ('left', 'center', 'right'), or 'invalid' if classification fails.
    """
    system_prompt = (
        "You are a helpful assistant that classifies news articles by bias. "
        "Given the article details, reply ONLY with one of the bias labels: 'left', 'center', or 'right'."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": article_text}
    ]
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=10,
            temperature=0
        )
        pred = response.choices[0].message.content.strip().lower()
        if pred not in {"left", "center", "right"}:
            pred = "invalid"
        return pred
    except Exception as e:
        print(f"Error during API call: {e}")
        return "invalid"

def run_inference(client, model_name, texts):
    """
    Run list of articles using the OpenAI model.

    Args:
        client: The authenticated OpenAI API client.
        model_name: The ID of the fine-tuned model.
        texts: List of article text info to classify.

    Returns:
        list: Model predictions.
    """
    predictions = []
    for i, text in enumerate(tqdm(texts, desc="Inference on unlabeled test set")):
        print(f"Processing test article {i + 1}/{len(texts)}")
        pred_label = get_prediction_for_text(client, model_name, text)
        predictions.append(pred_label)
    return predictions

def save_results_and_reports(texts, true_labels, predictions, save_dir="Analysis"):
    """
    Save evaluation results to disk and output classification metrics and plots.

    Args:
        texts: The article texts that were classified.
        true_labels: The ground truth labels.
        predictions: Predictions from the model.
        save_dir: Directory to save outputs. Defaults to "Analysis".

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    df_eval = pd.DataFrame({
        "article_text": texts,
        "ground_truth": true_labels,
        "prediction": predictions
    })
    csv_path = os.path.join(save_dir, "test_no_labels_predictions_vs_ground_truth_deep_learning.csv")
    df_eval.to_csv(csv_path, index=False)
    print(f"Saved predictions vs ground truth to {csv_path}")

    # Accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\nTest Set Accuracy: {accuracy:.4f}\n")

    # Classification report
    valid_biases = ["left", "center", "right"]
    report = classification_report(true_labels, predictions, labels=valid_biases, zero_division=0)
    print("Classification Report:")
    print(report)
    with open(os.path.join(save_dir, "Deep Learning Model Classification Report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix plot
    cm = confusion_matrix(true_labels, predictions, labels=valid_biases)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_biases)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: Test Set")
    plot_path = os.path.join(save_dir, "Deep Learning Model Confusion Matrix.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion Matrix Plot Saved to {plot_path}")