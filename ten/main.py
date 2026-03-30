md  # =========================
# STEP 1: Import Libraries
# =========================
from tensorflow.keras.layers import Input, Dense, Embedding, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.models import Model

# =========================
# STEP 2: Prepare Data (reuse from above)
# =========================
X_inp = input_sequences[:, :-1]
y_out = tf.keras.utils.to_categorical(input_sequences[:, -1], num_classes=total_words)

# =========================
# STEP 3: Positional Encoding (Simple)
# =========================
def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

# =========================
# STEP 4: Build Transformer
# =========================
seq_len = max_seq_len - 1
embed_dim = 64

inputs = Input(shape=(seq_len,))
x = Embedding(total_words, embed_dim)(inputs)

# Add positional encoding
x = x + positional_encoding(seq_len, embed_dim)

# Attention Layer
attn_output = MultiHeadAttention(num_heads=2, key_dim=embed_dim)(x, x)
x = LayerNormalization()(x + attn_output)

# Feed Forward
ff = Dense(128, activation='relu')(x)
ff = Dense(embed_dim)(ff)
x = LayerNormalization()(x + ff)

# Output
x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = Dense(total_words, activation='softmax')(x)

transformer_model = Model(inputs, outputs)

transformer_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
transformer_model.summary()

# =========================
# STEP 5: Train
# =========================
transformer_model.fit(X_inp, y_out, epochs=100, verbose=1)

# =========================
# STEP 6: Generate Text
# =========================
def generate_text_transformer(seed_text, next_words=5):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=seq_len, padding='pre')

        predicted = np.argmax(transformer_model.predict(token_list), axis=-1)

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break
    return seed_text

print(generate_text_transformer("machine learning", 5))