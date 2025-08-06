import json
import os
import time
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from openai import OpenAI
from tqdm import tqdm

def load_validation_data(filepath):
    """
    Load vtexts and ground truths from a .json file.

    Args:
        filepath: Path to the .json file containing validation data.

    Returns:
  
            texts: inputs for classification.
            true_labels: Ground-truth labels ('left', 'center', or 'right').
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line) for line in f]
    texts = []
    true_labels = []
    for item in val_data:
        user_message = next(m['content'] for m in item['messages'] if m['role'] == 'user')
        texts.append(user_message)
        assistant_message = next(m['content'] for m in item['messages'] if m['role'] == 'assistant')
        true_labels.append(assistant_message.strip().lower())
    print(f"Loaded {len(texts)} validation examples.")
    return texts, true_labels

def initialize_client():
    """
    Initialize and return an OpenAI API client.

    Returns:
        OpenAI: An OpenAI client instance.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    return client

def get_prediction(client, model_name, article_text):
    """
    Submit an article's text to an OpenAI model to predict the bias label.

    Args:
        client: OpenAI API client.
        model_name: The ID of the fine-tuned model.
        article_text: Text of the news article to classify.

    Returns:
        str: Model Predictions. May return 'invalid' if output does not match prompt.
    """
    system_prompt = (
        "You are a helpful assistant that classifies news articles by bias. "
        "Given the article details, reply ONLY with one of the bias labels: 'left', 'center', or 'right'."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": article_text}
    ]
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

def run_predictions_validation(client, model_name, texts, sleep_time=0.2):
    """
    Model predictions for all validation articles using the OpenAI model.

    Args:
        client: OpenAI API client.
        model: The ID of the model to use for predictions.
        texts: List of article texts to classify.
        sleep_time: limit queries to avoid rate limits. Defaults to 0.2.

    Returns:
        Predicted bias labels for each article.
    """
    predictions = []
    for idx, text in enumerate(tqdm(texts, desc="Running predictions")):
        print(f"Processing validation example {idx+1}/{len(texts)}...")
        pred_label = get_prediction(client, model_name, text)
        predictions.append(pred_label)
        time.sleep(sleep_time)  # Rate limit control
    return predictions

def save_evaluation_results_validation(texts, true_labels, predictions, save_dir="Analysis"):
    """
    Save validation predictions, classification report, and confusion matrix plot to files.

    Args:
        texts: List of validation article texts.
        true_labels: List of ground-truth labels.
        predictions: Model predictions for each article.
        save_dir: Directory to outputs. Defaults to "Analysis".

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame({
        "article": texts,
        "true_label": true_labels,
        "predicted_label": predictions
    })
    csv_path = os.path.join(save_dir, "deep_learning validation_predictions_vs_ground_truth.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved validation predictions to {csv_path}")

    accuracy = (df['true_label'] == df['predicted_label']).mean()
    print(f"\nValidation accuracy: {accuracy:.4f}")

    # Classification report
    report = classification_report(df['true_label'], df['predicted_label'], labels=["left", "center", "right"], zero_division=0)
    print("\nClassification report:")
    print(report)
    with open(os.path.join(save_dir, "Deep Learning Model Validation Classification Report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix plot
    cm = confusion_matrix(df['true_label'], df['predicted_label'], labels=["left", "center", "right"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["left", "center", "right"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Validation)")
    plot_path = os.path.join(save_dir, "Deep Learning Model Validation Confusion Matrix.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved confusion matrix plot to {plot_path}")