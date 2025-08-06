import pandas as pd
from naive_model import prepare_prompts, get_predictions, save_results, evaluate_and_plot
from svm  import load_and_clean, split_data, vectorize, tune_and_train_svm, save_svm_model 
from deep_learning_train  import load_and_shuffle_jsonl, write_jsonl, upload_file, start_fine_tuning 
from deep_learning_test_model import load_unlabeled_texts, load_ground_truth_labels, run_inference, save_results_and_reports
from validation_deep_learning import load_validation_data, initialize_client, run_predictions_validation, save_evaluation_results_validation 
import os
from openai import OpenAI

############ Naive Model Open AI
df = pd.read_csv("Data/Raw/raw_dataset.csv", encoding="latin1")
prompts = prepare_prompts(df)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
responses = get_predictions(prompts, client)
df['predicted_label'] = responses
save_results(df, filename="Data/Processed/naive_model_evaluation_dataset.csv")
true_labels = df['bias_rating']
predicted_labels = df['predicted_label']
evaluate_and_plot(true_labels, predicted_labels, save_dir="Analysis")
###########

############ Traditional Model Support Vector Machine (SVM)

df_svm = load_and_clean("Data/Raw/raw_dataset.csv")
train, val, test = split_data(df_svm)
X_train, X_val, X_test, vectorizer = vectorize(train, val, test)
y_train, y_val, y_test = train["bias_rating"], val["bias_rating"], test["bias_rating"]
svm_clf = tune_and_train_svm(X_train, y_train)
save_svm_model(svm_clf, X_val, y_val, X_test, y_test, save_dir="Analysis")

###########

############ Deep Learning Model Training Open AI

input_path = "Data/Processed/formatted_training_data.json2"
output_dir = "Data/Processed"
train_path = os.path.join(output_dir, "train.json3")
val_path = os.path.join(output_dir, "val.json3")
test_path = os.path.join(output_dir, "test.json3")

# Shuffle and split the data
data = load_and_shuffle_jsonl(input_path)
train_data, val_data, test_data = split_data(data)

# Save splits
write_jsonl(train_path, train_data)
write_jsonl(val_path, val_data)
write_jsonl(test_path, test_data)

# Initialize API 
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Upload files
train_file_id = upload_file(client, train_path)
val_file_id = upload_file(client, val_path)

# Start fine-tuning job
base_model = "gpt-4.1-mini-2025-04-14"  # replace with your fine-tune base model name
start_fine_tuning(client, train_file_id, val_file_id, base_model)

###########
############ Deep Learning Model Validation Open AI

val_filepath = "val.json3"
fine_tuned_model = "ft:gpt-4.1-mini-2025-04-14:duke-university::C0Gn9Szw"  # Adjust as needed

texts, true_labels = load_validation_data(val_filepath)
client = initialize_client()
predictions = run_predictions_validation(client, fine_tuned_model, texts)
save_evaluation_results_validation(texts, true_labels, predictions, save_dir="Analysis")

###########
############ Deep Learning Model Testing Open AI

unlabeled_path = "test_no_labels.json2"
labeled_path = "test.json3"

# Initialize API
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

fine_tuned_model = "ft:gpt-4.1-mini-2025-04-14:duke-university::C0Gn9Szw"  # replace with your model

texts = load_unlabeled_texts(unlabeled_path)
true_labels = load_ground_truth_labels(labeled_path)

if len(texts) != len(true_labels):
    raise ValueError(f"Unequal input and label count: {len(texts)} vs {len(true_labels)}")

predictions = run_inference(client, fine_tuned_model, texts)

# Save results
save_results_and_reports(texts, true_labels, predictions, save_dir="Analysis")
###########


