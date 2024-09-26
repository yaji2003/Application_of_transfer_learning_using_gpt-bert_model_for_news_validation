import os
import time
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import TFGPT2Model, GPT2Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Disable OneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


start_time = time.time()


# Load your dataset
df = pd.read_csv('Dataset/train10.csv')
# Remove lines where 'text' is an empty string
df = df[df['text'].str.strip() != '']
# Drop rows with missing values in the 'text' column
df = df.dropna(subset=['text'])


# Tokenize and pad sequences
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
max_len = 500  # Set your desired sequence length (reduce it)

# Tokenize and pad the input text
def tokenize_and_pad(text):
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_len, truncation=True)
    return pad_sequences([tokens], maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")[0]

# Apply tokenization and padding to the 'text' column
df['input_ids'] = df['text'].apply(tokenize_and_pad)

# Split the dataset
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert input and output data to numpy arrays
x_train = tf.convert_to_tensor(train_df['input_ids'].to_list())
y_train = tf.convert_to_tensor(train_df['label'].values)

x_val = tf.convert_to_tensor(val_df['input_ids'].to_list())
y_val = tf.convert_to_tensor(val_df['label'].values)

# Set mixed precision policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Define the model
model = TFGPT2Model.from_pretrained('gpt2', output_hidden_states=True, dtype=tf.float32)
input_layer = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32)
gpt2_output = model(input_layer)['last_hidden_state']

# Add your classification layer on top of the GPT-2 output
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(gpt2_output[:, 0, :])

# Compile the model
fake_news_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
fake_news_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Set up checkpoints to save the model during training
checkpoint_filepath = 'model_checkpoint.h5'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    save_best_only=True,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# Train the model with time tracking
batch_size = 4  # Reduce batch size
epochs = 1



fake_news_model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[model_checkpoint]
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Training took {elapsed_time} seconds')

# Save the model after training
fake_news_model.save('fake_news_model.h5')
