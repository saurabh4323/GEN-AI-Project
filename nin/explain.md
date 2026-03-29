# 🧠 Text Generation using LSTM & Transformer

This project demonstrates **text generation using two deep learning architectures**:

* 🔁 LSTM (Recurrent Neural Network)
* ⚡ Transformer (Attention-based model)

Both models are trained on a small custom dataset and generate new text sequences word-by-word.

---

## 📌 Overview

The goal of this project is to:

* Learn how sequence models work
* Compare traditional RNN-based models (LSTM) with modern Transformer models
* Generate meaningful text based on a seed input

---

## 📂 Project Structure

```
Component I   → LSTM Model
Component II  → Transformer Model
```

---

## 🧾 Dataset

A small custom dataset of sentences related to:

* Machine Learning
* Deep Learning
* Sequence Modeling

Example:

```
machine learning models learn patterns from data
sequence models process data step by step
...
```

---

## ⚙️ Common Preprocessing Steps

Both models follow the same pipeline:

### 1. Tokenization

* Convert text → numerical tokens using Keras Tokenizer

### 2. Sequence Creation

* Convert sentences into input sequences
* Example:

  ```
  machine → learning → models
  ```

  becomes multiple training samples

### 3. Padding

* Ensure all sequences are of equal length

### 4. Input / Output Split

* X → sequence input
* y → next word prediction (one-hot encoded)

---

## 🔁 Component I: LSTM Model

### 🧠 Architecture

* Embedding Layer
* LSTM Layer (100 units)
* Dense Softmax Output

### 📈 Training

* Loss: Categorical Crossentropy
* Optimizer: Adam
* Epochs: 200

### 💡 Purpose

LSTM captures **sequential dependencies** using memory cells and gates.

---

## ⚡ Component II: Transformer Model

### 🧠 Architecture

* Embedding Layer
* 2× Transformer Blocks:

  * Multi-Head Attention
  * Feed Forward Network
  * Layer Normalization
* Global Average Pooling
* Dense Output Layer

### 📈 Training

* Loss: Categorical Crossentropy
* Optimizer: Adam
* Epochs: 100

### 💡 Purpose

Transformer uses **self-attention** to capture relationships between all words in a sequence simultaneously.

---

## ✍️ Text Generation

Both models use the same approach:

1. Take a **seed text**
2. Predict next word
3. Append predicted word
4. Repeat

Example:

```
Input:  "machine learning"
Output: "machine learning models learn patterns from"
```

---

## 🔍 Key Differences

| Feature     | LSTM                | Transformer           |
| ----------- | ------------------- | --------------------- |
| Type        | Sequential          | Parallel (Attention)  |
| Memory      | Hidden state        | Attention mechanism   |
| Speed       | Slower              | Faster                |
| Performance | Good for small data | Better for large data |

---

## 🚀 Insights

* LSTM works well for sequential understanding but struggles with long dependencies
* Transformer handles long-range dependencies better using attention
* With small data, both models may generate limited or repetitive text

---

## ⚠️ Limitations

* Very small dataset → limited learning
* No validation/testing split
* No hyperparameter tuning
* Basic architecture (not production-level)

---

## 📌 Future Improvements

* Use larger datasets (e.g., Wikipedia, news)
* Add dropout and regularization
* Use pretrained embeddings (GloVe, Word2Vec)
* Implement GPT-style autoregressive Transformer

---

## 🧑‍💻 Tech Stack

* Python
* TensorFlow / Keras
* NumPy

---

## ✅ Conclusion

This project shows a **side-by-side comparison of LSTM and Transformer models** for text generation and helps understand the evolution from RNNs to attention-based architectures.

---

## 📎 Reference

Code provided in: 

---
