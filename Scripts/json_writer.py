

import pandas as pd
import ast
import json

df = pd.read_csv("Data/Raw/raw_dataset.csv", encoding="latin1")

def build_messages(row):
    # Handle tags
    tags_list = row['tags']
    if isinstance(tags_list, str):
        try:
            tags_list = ast.literal_eval(tags_list)
        except Exception:
            pass
    tags_joined = ', '.join(tags_list) if isinstance(tags_list, (list, tuple)) else str(tags_list)
    
    # Compose messages
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that classifies news articles by bias. "
                    "Given the article details, reply ONLY with the bias label ('left', 'center', 'right')."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Title: {row['title']}\n"
                    f"Tags: {tags_joined}\n"
                    f"Heading: {row['heading']}\n"
                    f"Source: {row['source']}\n"
                    f"Text: {row['text']}\n"
                )
            },
            {
                "role": "assistant",
                "content": row['bias_rating'].strip()
            }
        ]
    }


# Write to JSONL
with open('formatted_training_data.json2', 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        json.dump(build_messages(row), f)
        f.write('\n')