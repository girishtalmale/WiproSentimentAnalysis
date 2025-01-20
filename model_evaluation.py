# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:54:03 2024

@author: GHRCE
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results from the training script
file_path = 'D:/Wipro/Wipro Advance AIML/Project 20/data/model_results.csv'  # Update to the location of your saved results
results_df = pd.read_csv(file_path)

# Display the results in a table format
print("Model Evaluation Results:")
print(results_df)

# Plot accuracy and F1-score for all models
plt.figure(figsize=(16, 12))

# Accuracy plot
plt.subplot(2, 2, 1)
sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)

# F1-Score plot
plt.subplot(2, 2, 2)
sns.barplot(x='Model', y='F1-Score', data=results_df, palette='viridis')
plt.title('Model F1-Score Comparison')
plt.xlabel('Model')
plt.ylabel('F1-Score')
plt.xticks(rotation=45)

# Precision plot
plt.subplot(2, 2, 3)
sns.barplot(x='Model', y='Precision', data=results_df, palette='viridis')
plt.title('Model Precision Comparison')
plt.xlabel('Model')
plt.ylabel('Precision')
plt.xticks(rotation=45)

# Recall plot
plt.subplot(2, 2, 4)
sns.barplot(x='Model', y='Recall', data=results_df, palette='viridis')
plt.title('Model Recall Comparison')
plt.xlabel('Model')
plt.ylabel('Recall')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('D:/Wipro/Wipro Advance AIML/Project 6/data/model_comparative_analysis.png')
plt.show()

# Find the best-performing model
best_model = results_df.loc[results_df['Accuracy'].idxmax()]
print("\nBest Performing Model:")
print(f"Model: {best_model['Model']}")
print(f"Accuracy: {best_model['Accuracy']}")
print(f"F1-Score: {best_model['F1-Score']}")
# print(f"Precision: {best_model['Precision']}")
# print(f"Recall: {best_model['Recall']}")
# print(f"Accuracy: {best_model['Accuracy']:.2f}")
# print(f"F1-Score: {best_model['F1-Score']:.2f}")
# print(f"Precision: {best_model['Precision']:.2f}")
# print(f"Recall: {best_model['Recall']:.2f}")

# Highlight areas for improvement
print("\nEvaluation Summary:")
for _, row in results_df.iterrows():
    print(f"{row['Model']}:")
    if row['Accuracy'] < 0.7:
        print("  - Accuracy is below 70%. Consider improving feature engineering or model hyperparameters.")
    if row['F1-Score'] < 0.7:
        print("  - F1-Score is below 70%. Look into handling class imbalance or using a different model.")

