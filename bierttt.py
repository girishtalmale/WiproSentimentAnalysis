# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:53:51 2024

@author: GHRCE
"""

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import BertTokenizer, TFBertForSequenceClassification
from tf_keras.optimizers.legacy import Adam


# Load preprocessed dataset
file_path = 'D:\Wipro\Wipro Advance AIML\Project 20/data/processed_survey.csv'
df = pd.read_csv(file_path)

# Prepare data
X = df['clean_comment'].values
y = df['sentiment_label'].values  # Ensure this is numeric (e.g., 0, 1, 2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize text data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(X_train), padding=True, truncation=True, max_length=128, return_tensors="tf")
test_encodings = tokenizer(list(X_test), padding=True, truncation=True, max_length=128, return_tensors="tf")

# Convert labels to TensorFlow tensors
train_labels = tf.convert_to_tensor(y_train)
test_labels = tf.convert_to_tensor(y_test)

# Load the pre-trained BERT model for classification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Use legacy Adam optimizer for compatibility
from tensorflow.keras.optimizers.legacy import Adam
optimizer = Adam(learning_rate=5e-5)

# Compile the model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Prepare datasets for training
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels)).batch(16)

# Train the model
print("\nTraining the BERT model...")
model.fit(train_dataset, validation_data=test_dataset, epochs=3)

# Evaluate the model
print("\nEvaluating the model...")
predictions = model.predict(test_dataset).logits
y_pred = tf.argmax(predictions, axis=1).numpy()

# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")

# Save the fine-tuned model
model.save_pretrained('models/bert_model/')
tokenizer.save_pretrained('models/bert_model/')
print("\nModel and tokenizer saved to 'models/bert_model/'.")
