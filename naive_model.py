import os
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def prepare_prompts(df):
    """
    Generate instructional prompts for news article bias classification.

    Args:
        df: DataFrame containing news article data with columns such as
            'title', 'tags', 'heading', 'source', 'text', and 'bias_rating'.

    Returns:
        list of str: A list of formatted prompts, each describing a news article's content
            and instructing the assistant to classify its bias as 'left', 'center', or 'right'.
    """
    input_data = df.drop(columns=["bias_rating"])
    prompts = []
    for _, row in input_data.iterrows():
        prompt = f"""You are a helpful assistant that classifies news articles by bias. Given the article details, reply ONLY with one of the bias labels: 'left', 'center', or 'right'.

Title: {row['title']}
Tags: {row['tags']}
Heading: {row['heading']}
Source: {row['source']}
Text: {row['text']}


"""
        prompts.append(prompt)
    return prompts

def get_predictions(prompts, client):
    """
    Predictions from naive OpenAI language model for given prompt.

    Args:
        prompts (list of str): List of prompts describing news articles to be classified.
        client (OpenAI): OpenAI API client for accessing chat completions.

    Returns:
        list of str: Model predictions for each prompt, typically one of 'left', 'center', 'right',
            or 'error' if the prediction fails.
    """
    responses = []
    for i, prompt in enumerate(tqdm(prompts, desc="Calling OpenAI API"), start=1):
        print(f"Starting New Prompt: {i}")
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32000,
                temperature=0
            )
            prediction_raw = response.choices[0].message.content.strip()
            try:
                prediction_json = json.loads(prediction_raw)
                prediction = prediction_json if isinstance(prediction_json, str) else prediction_json.get('label', prediction_raw)
            except json.JSONDecodeError:
                prediction = prediction_raw
            responses.append(prediction)
        except Exception as e:
            print(f"Error on prompt: {e}")
            responses.append("error")
    return responses

def save_results(df, filename):
    """
    Save the DataFrame containing articles, labels, and predictions to a CSV file.

    Args:
        df: DataFrame containing relevant results to save.
        filename (str): Path of location to CSV.

    Returns:
        None
    """
    df.to_csv(filename, index=False)

def evaluate_and_plot(true_labels, predicted_labels, save_dir="Analysis"):
    """
    Computing/saving classification metrics and confusion matrix for predictions from naive model.

    Args:
        true_labels: The ground-truth bias labels.
        predicted_labels: The predicted bias labels.
        save_dir: Directory where the classification report and confusion matrix plot
            will be saved. 

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    # Classification Report
    report = classification_report(true_labels, predicted_labels, target_names=['left', 'center', 'right'])
    print("Classification Report: Naive Model\n")
    print(report)
    with open(os.path.join(save_dir, "Naive Model Classification Report.txt"), "w") as f:
        f.write(report)
    # Confusion Matrix
    labels = ['left', 'center', 'right']
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: Bias Classification Naive Model")
    plot_path = os.path.join(save_dir, "Naive Model Confusion Matrix.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")