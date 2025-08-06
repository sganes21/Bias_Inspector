import os
import json
import random
from openai import OpenAI

def load_and_shuffle_jsonl(filepath, seed=0):
    """
    Load a .json file, parse each line as a json object, and shuffle the resulting list.

    Args:
        filepath: Input .json file.
        seed: Seed value for the random shuffle to ensure reproducibility. Defaults to 0.

    Returns:
        list: A shuffled list of .json objects loaded from the file.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    random.seed(seed)
    random.shuffle(data)
    return data

def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """
    Split dataset into training, validation, and test subsets.

    Args:
        data: Input data.
        train_ratio: Defaults to 0.8.
        val_ratio:  Defaults to 0.1.

    Returns:
        train, validation, test (lists).
    """
    N = len(data)
    train_end = int(train_ratio * N)
    val_end = int((train_ratio + val_ratio) * N)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data

def write_jsonl(filepath, dataset):
    """
    Write a list to  a .json file.

    Args:
        filepath: Output path for the .json file.
        dataset: List of objects to be written.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in dataset:
            json.dump(item, f)
            f.write('\n')
    print(f"Saved {len(dataset)} records to {filepath}")

def upload_file(client, filepath, purpose="fine-tune"):
    """
    Upload a file to the OpenAI API to fine tune model.

    Args:
        client: OpenAPI.
        filepath: Path to the file to upload.
        purpose:  Defaults to "fine-tune".

    Returns:
        str: The OpenAI file ID of the uploaded file.
    """
    with open(filepath, "rb") as f:
        resp = client.files.create(file=f, purpose=purpose)
    print(f"Uploaded {filepath} with id {resp.id}")
    return resp.id

def start_fine_tuning(client, training_file_id, validation_file_id, base_model):
    """
    Start a fine-tuning job on OpenAI with specified training and validation file IDs and a base model.

    Args:
        client: OpenAI API client instance.
        training_file_id: file ID for the training set.
        validation_file_id: file ID for the validation set.
        base_model: Name of the base model to fine-tune.

    Returns:
        object: The response object returned by the OpenAI API for the fine-tuning job creation.
    """
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=base_model,
        hyperparameters={
            "n_epochs": 4,
            "learning_rate_multiplier": 0.1,
            # add other hyperparameters if desired
        }
    )
    print("Fine-tuning job started:", response)
    return response