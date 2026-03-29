

# =============================
# COMPONENT I: LSTM MODEL
# =============================

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# -----------------------------
# DATASET
# -----------------------------
data = """machine learning models learn patterns from data
sequence models process data step by step
recurrent neural networks are designed for sequential tasks
rnn models maintain hidden states across time steps
long short term memory networks solve long dependency problems
lstm uses gates to control information flow
gru models simplify the lstm architecture
sequence prediction is useful in many applications
language modeling predicts the next word in a sentence
speech recognition processes audio sequences
time series forecasting predicts future values
music generation creates new melodies
generative models learn probability distributions
they generate new samples similar to training data
sequence generation is widely used in artificial intelligence
deep learning improves sequence modeling performance"""

# -----------------------------
# TOKENIZATION
# -----------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
total_words = len(tokenizer.word_index) + 1

# -----------------------------
# CREATE SEQUENCES
# -----------------------------
input_sequences = []

for line in data.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

max_len = max(len(x) for x in input_sequences)

input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# -----------------------------
# MODEL
# -----------------------------
model = Sequential([
    Embedding(total_words, 64, input_length=max_len-1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# -----------------------------
# TRAIN
# -----------------------------
model.fit(X, y, epochs=200, verbose=1)

# -----------------------------
# GENERATE TEXT
# -----------------------------
def generate_text(seed_text, next_words=5):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

        predicted = np.argmax(model.predict(token_list), axis=-1)

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break

    return seed_text

print(generate_text("machine learning", 5))
     

# =============================
# COMPONENT II: TRANSFORMER MODEL
# =============================

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dropout, Dense, Input, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Model

# -----------------------------
# DATASET
# -----------------------------
data = """machine learning models learn patterns from data
sequence models process data step by step
recurrent neural networks are designed for sequential tasks
rnn models maintain hidden states across time steps
long short term memory networks solve long dependency problems
lstm uses gates to control information flow
gru models simplify the lstm architecture
sequence prediction is useful in many applications
language modeling predicts the next word in a sentence
speech recognition processes audio sequences
time series forecasting predicts future values
music generation creates new melodies
generative models learn probability distributions
they generate new samples similar to training data
sequence generation is widely used in artificial intelligence
deep learning improves sequence modeling performance"""

# -----------------------------
# TOKENIZATION
# -----------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
total_words = len(tokenizer.word_index) + 1

# -----------------------------
# CREATE SEQUENCES
# -----------------------------
input_sequences = []

for line in data.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

max_len = max(len(x) for x in input_sequences)

input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# -----------------------------
# TRANSFORMER BLOCK
# -----------------------------
def transformer_block(x):
    attn = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    attn = Dropout(0.1)(attn)
    out1 = LayerNormalization()(x + attn)

    ffn = Dense(64, activation='relu')(out1)
    ffn = Dense(64)(ffn)
    return LayerNormalization()(out1 + ffn)

# -----------------------------
# MODEL
# -----------------------------
inputs = Input(shape=(max_len-1,))
x = Embedding(total_words, 64)(inputs)

x = transformer_block(x)
x = transformer_block(x)

x = GlobalAveragePooling1D()(x)
outputs = Dense(total_words, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# -----------------------------
# TRAIN
# -----------------------------
model.fit(X, y, epochs=100, verbose=1)

# -----------------------------
# GENERATE TEXT
# -----------------------------
def generate_text(seed_text, next_words=5):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

        predicted = np.argmax(model.predict(token_list), axis=-1)

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break

    return seed_text

print(generate_text("deep learning", 5))
   