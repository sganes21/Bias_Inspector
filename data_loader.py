import pandas as pd
import json
import os
from tqdm import tqdm 
from openai import OpenAI

# Loading
df = pd.read_csv("sample_dataset.csv", encoding="latin1")
input_data = df.drop(columns=["bias_rating"])

# Gathering inputs into model
prompts = []
for _, row in input_data.iterrows():
    prompt = f"""Given the following news article information, predict its political bias (left, center, or right) and its severity on a scale from 0 to 1:

Title: {row['title']}
Tags: {row['tags']}
Heading: {row['heading']}
Source: {row['source']}
Text: {row['text']}

Just provide an output like this: 
{{
  "bias":"left",
  "severity":0.32
}}
"""
    prompts.append(prompt)

# API Key

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Outputs
responses = []
#for prompt in prompts:
#for prompt in tqdm(prompts, desc="Calling OpenAI API"):
for i, prompt in enumerate(tqdm(prompts, desc="Calling OpenAI API"), start=1):
    print(f"Starting New Prompt: {i}")
    try:
        response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32000,
        temperature=0
    )
    #print(response.choices[0].message.content.strip())
        prediction = json.loads(response.choices[0].message.content.strip())
        responses.append(prediction)
    except Exception as e:
        print(f"Error on prompt: {e}")
   
# Save results
df = pd.DataFrame(responses)
df.to_csv("predictions.csv", index=False)