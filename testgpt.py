import os
import time
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import TFGPT2Model, GPT2Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Disable OneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load your dataset
df = pd.read_csv('Dataset/test4.csv') 
df = df[df['text'].str.strip() != '']
df = df.dropna(subset=['text'])

# Tokenize and pad sequences
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
max_len = 500

def tokenize_and_pad(text):
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_len, truncation=True)
    return pad_sequences([tokens], maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")[0]

df['input_ids'] = df['text'].apply(tokenize_and_pad)

# Split the dataset
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert input and output data to numpy arrays
x_val_tokens = val_df['text'].apply(tokenize_and_pad)
x_val_tokens = tf.convert_to_tensor(x_val_tokens.to_list())
y_val = tf.convert_to_tensor(val_df['label'].values)

# Define custom_objects for loading the model
custom_objects = {
    'TFGPT2Model': TFGPT2Model,
    'tokenizer': tokenizer,  # You may need to include other custom objects used in the model
}

# Load the saved model
loaded_model = tf.keras.models.load_model('fake_news_model.h5', custom_objects=custom_objects)

# Predict on the tokenized validation set
y_val_pred = loaded_model.predict(x_val_tokens)

# Apply threshold for binary classification
threshold = 0.5
y_val_pred_binary = (y_val_pred[:, 0] > threshold).astype(int)

# Evaluate accuracy
accuracy = accuracy_score(y_val, y_val_pred_binary)
print(f'Accuracy: {accuracy}')

# Generate a classification report
classification_rep = classification_report(y_val, y_val_pred_binary)
print('Classification Report:\n', classification_rep)

# Generate a confusion matrix
confusion_mat = confusion_matrix(y_val, y_val_pred_binary)
print('Confusion Matrix:\n', confusion_mat)
