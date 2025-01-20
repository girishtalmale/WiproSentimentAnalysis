# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:52:39 2024

@author: GHRCE
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Load preprocessed dataset
file_path = 'D:\Wipro\Wipro Advance AIML\Project 20/data/processed_survey.csv'  # Update with your preprocessed dataset's file path
df = pd.read_csv(file_path)

# Split data into training and testing sets
X = df['clean_comment']
y = df['sentiment_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize results storage
results = []

# ---- Model 1: Naive Bayes ----
print("\nTraining Naive Bayes...")
vectorizer_nb = CountVectorizer()
X_train_nb = vectorizer_nb.fit_transform(X_train)
X_test_nb = vectorizer_nb.transform(X_test)

model_nb = MultinomialNB()
model_nb.fit(X_train_nb, y_train)
y_pred_nb = model_nb.predict(X_test_nb)

# results.append({
#     'Model': 'Naive Bayes',
#     'Accuracy': accuracy_score(y_test, y_pred_nb),
#     'F1-Score': f1_score(y_test, y_pred_nb, average='weighted')
# })

results.append({
    'Model': 'Naive Bayes',
    'Accuracy': accuracy_score(y_test, y_pred_nb),
    'F1-Score': f1_score(y_test, y_pred_nb, average='weighted'),
    'Precision': precision_score(y_test, y_pred_nb, average='weighted'),
    'Recall': recall_score(y_test, y_pred_nb, average='weighted')
})
print(classification_report(y_test, y_pred_nb))

# ---- Model 2: Random Forest ----
print("\nTraining Random Forest...")
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train_nb, y_train)
y_pred_rf = model_rf.predict(X_test_nb)

# results.append({
#     'Model': 'Random Forest',
#     'Accuracy': accuracy_score(y_test, y_pred_rf),
#     'F1-Score': f1_score(y_test, y_pred_rf, average='weighted')
# })

results.append({
    'Model': 'Random Forest',
    'Accuracy': accuracy_score(y_test, y_pred_nb),
    'F1-Score': f1_score(y_test, y_pred_nb, average='weighted'),
    'Precision': precision_score(y_test, y_pred_nb, average='weighted'),
    'Recall': recall_score(y_test, y_pred_nb, average='weighted')
})
print(classification_report(y_test, y_pred_rf))

# ---- Model 3: LSTM ----
print("\nTraining LSTM...")
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=100)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

model_lstm = Sequential([
    Embedding(input_dim=5000, output_dim=64),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(3, activation='softmax')
])
model_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_train_pad, y_train, epochs=5, batch_size=32, verbose=1)

y_pred_lstm = model_lstm.predict(X_test_pad).argmax(axis=1)

# results.append({
#     'Model': 'LSTM',
#     'Accuracy': accuracy_score(y_test, y_pred_lstm),
#     'F1-Score': f1_score(y_test, y_pred_lstm, average='weighted')
# })
results.append({
    'Model': 'LSTM',
    'Accuracy': accuracy_score(y_test, y_pred_nb),
    'F1-Score': f1_score(y_test, y_pred_nb, average='weighted'),
    'Precision': precision_score(y_test, y_pred_nb, average='weighted'),
    'Recall': recall_score(y_test, y_pred_nb, average='weighted')
})
print(classification_report(y_test, y_pred_lstm))

# ---- Model 4: BERT ----


# import tensorflow as tf
# from transformers import BertTokenizer, TFBertForSequenceClassification
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score, f1_score

# # Load data and tokenizer
# file_path = 'D:\Wipro\mainnnn student sentiment project\Project 18/data/processed_survey.csv'  # Replace with your file path
# df = pd.read_csv(file_path)
# X = df['clean_comment'].values
# y = df['sentiment_label'].values

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Tokenize the data
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# train_encodings = tokenizer(list(X_train), padding=True, truncation=True, max_length=128, return_tensors="tf")
# test_encodings = tokenizer(list(X_test), padding=True, truncation=True, max_length=128, return_tensors="tf")

# # Convert to tensors
# train_labels = tf.convert_to_tensor(y_train)
# test_labels = tf.convert_to_tensor(y_test)

# # Load the model
# model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# # Set up optimizer and loss
# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# # Create training loop
# for epoch in range(3):
#     print(f'Epoch {epoch + 1}')
#     for i in range(0, len(X_train), 16):  # 16 is the batch size
#         # Get a batch of data
#         input_ids = train_encodings['input_ids'][i:i + 16]
#         attention_mask = train_encodings['attention_mask'][i:i + 16]
#         batch_labels = train_labels[i:i + 16]

#         with tf.GradientTape() as tape:
#             outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)
#             loss_value = outputs.loss

#         grads = tape.gradient(loss_value, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))

#     # Evaluate the model on test data
#     test_output = model(test_encodings['input_ids'], attention_mask=test_encodings['attention_mask'], labels=test_labels)
#     test_loss, test_accuracy = test_output.loss, test_output.logits

#     # Print the test accuracy and loss for each epoch
#     print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy.numpy().mean()}")

# # Final evaluation
# predictions = model.predict(test_encodings['input_ids'], attention_mask=test_encodings['attention_mask']).logits
# y_pred = tf.argmax(predictions, axis=1).numpy()

# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred, average='weighted')

# print("Classification Report:")
# print(classification_report(y_test, y_pred))
# print(f"Accuracy: {accuracy}")
# print(f"F1-Score: {f1}")


# Save Results
results_df = pd.DataFrame(results)
results_df.to_csv('data/model_results.csv', index=False)
print("\nModel training completed. Results saved to 'data/model_results.csv'.")
