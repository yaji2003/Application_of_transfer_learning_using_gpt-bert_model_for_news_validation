import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
import os

# Load and preprocess the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# BERT Model
def create_fine_tuned_bert_model(model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Make BERT layers trainable
    model.trainable = True

    # Introduce dropout in BERT model
    model.bert.pooler.dropout = Dropout(0.1)

    return model, tokenizer

# Train the model
def train_model(model, train_data, tokenizer, model_save_path):
    train_data['text'] = train_data['text'].astype(str)
    input_ids = tokenizer(train_data['text'].tolist(), padding=True, truncation=True, return_tensors='tf')['input_ids']
    input_ids_numpy = input_ids.numpy()
    train_data['label'] = train_data['label'].fillna(0)
    train_data['label'] = pd.to_numeric(train_data['label'], errors='coerce')
    train_data['label'] = train_data['label'].fillna(0)
    labels = train_data['label'].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(input_ids_numpy, labels, test_size=0.2, random_state=42)

    # Print labels in train data
    print("Labels in Train Data:")
    print(labels)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=3,  # Increase the number of epochs
        batch_size=16,
        callbacks=[early_stopping]
    )

    # Save the updated model and tokenizer
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

if __name__ == "__main__":
    # Load data
    train_data = load_data('Dataset/train10.csv')

    # Define the path to save the model
    model_save_path = 'fine_tuned_bert_model'

    # Check if the model exists, if not, create a new one
    if not os.path.exists(model_save_path):
        bert_model, bert_tokenizer = create_fine_tuned_bert_model()
    else:
        bert_model = TFBertForSequenceClassification.from_pretrained(model_save_path)
        bert_tokenizer = BertTokenizer.from_pretrained(model_save_path)

    # Train the model
    train_model(bert_model, train_data, bert_tokenizer, model_save_path)
