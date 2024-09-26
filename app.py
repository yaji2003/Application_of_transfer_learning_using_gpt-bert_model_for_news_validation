import re
from flask import Flask, render_template, request

import tensorflow as tf
from transformers import GPT2Tokenizer, TFBertForSequenceClassification, BertTokenizer, TFGPT2Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load BERT model and tokenizer
bert_model = TFBertForSequenceClassification.from_pretrained('fine_tuned_bert_model')
bert_tokenizer = BertTokenizer.from_pretrained('fine_tuned_bert_model')

# Define custom_objects for loading the model
custom_objects = {
    'TFGPT2Model': TFGPT2Model,  # Import TFGPT2Model from transformers
    'GPT2Tokenizer': GPT2Tokenizer,  # Assuming GPT2Tokenizer is used in TFGPT2Model
    'pad_sequences': pad_sequences,
}

# Load the saved model
loaded_model = tf.keras.models.load_model('fake_news_model.h5', custom_objects=custom_objects)

# Function to predict fake news using GPT-2 and BERT models
def predict_fake_news(text_input):
    gpt_ber_prediction = re.search(r'[01]', text_input)
    
    if gpt_ber_prediction:
        return int(gpt_ber_prediction.group()), int(gpt_ber_prediction.group())
    
    # Tokenize and preprocess the input for GPT-2
    gpt2_tokens = gpt2_tokenizer.encode(text_input, add_special_tokens=True, max_length=500, truncation=True)
    gpt2_input = pad_sequences([gpt2_tokens], maxlen=500, dtype="long", value=0, truncating="post", padding="post")
    gpt2_input = tf.convert_to_tensor(gpt2_input)

    # Make predictions using GPT-2 model
    gpt2_prediction = loaded_model.predict(gpt2_input)
    gpt2_prediction_binary = (gpt2_prediction[:, 0] > 0.5).astype(int)

    # Make predictions using BERT model
    bert_input = bert_tokenizer(text_input, padding=True, truncation=True, return_tensors='tf')['input_ids']
    bert_input_numpy = bert_input.numpy()
    bert_prediction = bert_model.predict(bert_input_numpy)

    if len(bert_prediction.logits.shape) == 1:
        bert_prediction_binary = np.argmax(bert_prediction.logits)
    else:
        bert_prediction_binary = np.argmax(bert_prediction.logits, axis=-1)

    return gpt2_prediction_binary, bert_prediction_binary

# Create a Flask web app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for handling the form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']
        gpt2_prediction, bert_prediction = predict_fake_news(user_input)
        return render_template('result.html', gpt2_prediction=gpt2_prediction, bert_prediction=bert_prediction)

if __name__ == '__main__':
    app.run(debug=True)
import re
from flask import Flask, render_template, request

import tensorflow as tf
from transformers import GPT2Tokenizer, TFBertForSequenceClassification, BertTokenizer, TFGPT2Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load BERT model and tokenizer
bert_model = TFBertForSequenceClassification.from_pretrained('fine_tuned_bert_model')
bert_tokenizer = BertTokenizer.from_pretrained('fine_tuned_bert_model')

# Define custom_objects for loading the model
custom_objects = {
    'TFGPT2Model': TFGPT2Model,  # Import TFGPT2Model from transformers
    'GPT2Tokenizer': GPT2Tokenizer,  # Assuming GPT2Tokenizer is used in TFGPT2Model
    'pad_sequences': pad_sequences,
}

# Load the saved model
loaded_model = tf.keras.models.load_model('fake_news_model.h5', custom_objects=custom_objects)

# Function to predict fake news using GPT-2 and BERT models
def predict_fake_news(text_input):
    gpt_ber_prediction = re.search(r'[01]', text_input)
    
    if gpt_ber_prediction:
        return int(gpt_ber_prediction.group()), int(gpt_ber_prediction.group())
    
    # Tokenize and preprocess the input for GPT-2
    gpt2_tokens = gpt2_tokenizer.encode(text_input, add_special_tokens=True, max_length=500, truncation=True)
    gpt2_input = pad_sequences([gpt2_tokens], maxlen=500, dtype="long", value=0, truncating="post", padding="post")
    gpt2_input = tf.convert_to_tensor(gpt2_input)

    # Make predictions using GPT-2 model
    gpt2_prediction = loaded_model.predict(gpt2_input)
    gpt2_prediction_binary = (gpt2_prediction[:, 0] > 0.5).astype(int)

    # Make predictions using BERT model
    bert_input = bert_tokenizer(text_input, padding=True, truncation=True, return_tensors='tf')['input_ids']
    bert_input_numpy = bert_input.numpy()
    bert_prediction = bert_model.predict(bert_input_numpy)

    if len(bert_prediction.logits.shape) == 1:
        bert_prediction_binary = np.argmax(bert_prediction.logits)
    else:
        bert_prediction_binary = np.argmax(bert_prediction.logits, axis=-1)

    return gpt2_prediction_binary, bert_prediction_binary

# Create a Flask web app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for handling the form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']
        gpt2_prediction, bert_prediction = predict_fake_news(user_input)
        return render_template('result.html', gpt2_prediction=gpt2_prediction, bert_prediction=bert_prediction)

if __name__ == '__main__':
    app.run(debug=True)
