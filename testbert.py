import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Load and preprocess the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Evaluate the model
def evaluate_model(model, test_data, tokenizer):
    test_data['text'] = test_data['text'].astype(str)
    input_ids = tokenizer(test_data['text'].tolist(), padding=True, truncation=True, return_tensors='tf')['input_ids']
    input_ids_numpy = input_ids.numpy()
    test_data['label'] = test_data['label'].fillna(0)
    test_data['label'] = pd.to_numeric(test_data['label'], errors='coerce')
    test_data['label'] = test_data['label'].fillna(0)
    labels = test_data['label'].astype(int)

    # Make predictions
    predictions = model.predict(input_ids_numpy)

    # Get predicted labels using argmax
    if len(predictions.logits.shape) == 1:
        predicted_labels = tf.argmax(predictions.logits).numpy()
    else:
        predicted_labels = tf.argmax(predictions.logits, axis=-1).numpy()

    # Calculate confusion matrix, accuracy, precision, recall, and F1 score
    cm = confusion_matrix(labels, predicted_labels)
    accuracy = accuracy_score(labels, predicted_labels)
    report = classification_report(labels, predicted_labels)

    print(f'Confusion Matrix:\n{cm}')
    print(f'Accuracy: {accuracy}')
    print(f'Classification Report:\n{report}')

    return accuracy, report

if __name__ == "__main__":
    # Load data
    test_data = load_data('Dataset/test4.csv')

    # Load the model and tokenizer
    loaded_model = TFBertForSequenceClassification.from_pretrained('fine_tuned_bert_model')
    tokenizer = BertTokenizer.from_pretrained('fine_tuned_bert_model')

    # Evaluate the model
    accuracy, report = evaluate_model(loaded_model, test_data, tokenizer)
