import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("sample_dataset.csv", encoding="latin1")
#df = pd.read_excel("sample_snippet.xlsx")

#df['input_text'] = df['title'] + " " + df['text']

# Fill missing text fields
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')
df['input_text'] = df['title'] + " " + df['text']

# Remove rows without a bias label (if any)
df = df.dropna(subset=['bias_rating'])

# Initial train-test split (80% train+val, 20% test)
train_val, test = train_test_split(
    df, test_size=0.2, random_state=0, stratify=df['bias_rating']
)
# Further split training set into training and validation (75%/25% of train+val)
train, val = train_test_split(
    train_val, test_size=0.25, random_state=0, stratify=train_val['bias_rating']
)


# Create vectorizer and fit on training data
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train['input_text'])
X_val = vectorizer.transform(val['input_text'])
X_test = vectorizer.transform(test['input_text'])

# Get labels
y_train = train['bias_rating']
y_val = val['bias_rating']
y_test = test['bias_rating']

# Train SVM (linear kernel is standard for text)
svm_clf = LinearSVC(random_state=0)
svm_clf.fit(X_train, y_train)

# Predict on validation set for tuning (optional)
val_preds = svm_clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))

# Predict on test set
test_preds = svm_clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_preds))

print("Classification Report:\n", classification_report(y_test, test_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds))



# def load_image(path):
#     img = Image.open(path).convert('L')  
    
#     # Check if image is within specified range
#     width, height = img.size
#     if not (32 <= width <= 500 and 32 <= height <= 500):
#         raise ValueError(f"Image size must be between 32x32 and 500x500. Current size: {width}x{height}")
#     img_array = np.array(img)  
#     return img_array.flatten()  

# def svm_model_train(data, labels):
    
#     from sklearn.svm import SVC
#     svm_model = SVC()
#     svm_model.fit(data, labels)
#     return svm_model
    

# def svm_model_prediction(svm_model,new_data):

#     prediction= svm_model.predict(new_data)
#     return prediction

# def load_dataset(data_dir, categories):
   
#     data = []
#     labels = []
#     filenames = []
    
#     for category_idx, category in enumerate(categories):
#         category_path = os.path.join(data_dir, category)
#         if not os.path.exists(category_path):
#             print(f"Warning: Category path {category_path} does not exist. Skipping.")
#             continue
            
#         print(f"Loading {category} images...")
#         image_paths = glob.glob(os.path.join(category_path, "*.jpg")) + \
#                      glob.glob(os.path.join(category_path, "*.png"))
        
#         for image_path in image_paths:
#             try:
  
#                 img_features = load_image(image_path)
#                 data.append(img_features)
#                 labels.append(category_idx)
#                 filenames.append(os.path.basename(image_path))
#             except Exception as e:
#                 print(f"Error loading {image_path}: {e}")
                
#     return np.array(data), np.array(labels), filenames


# def train_validate_model(data_dir, categories, test_size=0.2, random_state=0, save_model_path=None):

#     # Load dataset
#     X, y, filenames = load_dataset(data_dir, categories)
    
#     if len(X) == 0:
#         print("No valid images found. Please double-checkcheck data directory.")
#         return None
    
#     print(f"Loaded {len(X)} images with shape: {X[0].shape}")
    

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
#     )
    
#     print(f"Training set: {X_train.shape[0]} samples")
#     print(f"Validation set: {X_val.shape[0]} samples")
    
#     # Train/validate SVM 
#     print("Training SVM model...")
#     model = svm_model_train(X_train, y_train)
#     print("Validating model...")
#     y_pred = svm_model_prediction(model, X_val)
    
#     # Calculate best fit 
#     accuracy = accuracy_score(y_val, y_pred)
#     print(f"Validation accuracy: {accuracy:.4f}")
#     print("\nClassification Report:")
#     print(classification_report(y_val, y_pred, target_names=categories))
    
#     if save_model_path:
#         os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
#         model_data = {
#             'model': model,
#             'scaler': scaler,
#             'categories': categories
#         }
#         joblib.dump(model_data, save_model_path)
#         print(f"Model saved to {save_model_path}")
    
#     return model, scaler

# def predict_on_new_images(model, scaler, categories, image_paths):

#     predictions = []
#     for image_path in image_paths:
#         try:
         
#             img_features = load_image(image_path)
#             img_features_scaled = scaler.transform([img_features])
            
#             # Predictions
#             prediction = svm_model_prediction(model, img_features_scaled)[0]
#             predicted_category = categories[prediction]
            
#             predictions.append({
#                 'image': image_path,
#                 'prediction': predicted_category,
#                 'prediction_idx': prediction
#             })
            
#             # Display image and prediction
#             img = plt.imread(image_path)
#             plt.figure(figsize=(6, 6))
#             plt.imshow(img, cmap='gray')
#             plt.title(f"THe predicted class is: {predicted_category}")
#             plt.axis('off')
#             plt.show()
            
#         except Exception as e:
#             print(f"Error processing {image_path}: {e}")
    
#     return predictions



