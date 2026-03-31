# Transformer_1
A decoder only transformer architecture based model :
# 🔮 Decoder-Only Transformer for Sine Wave Generation

This project implements a **decoder-only Transformer model from scratch** using PyTorch and applies it to **time-series prediction** — specifically, generating a sine wave.

---

## 🧠 Motivation

Transformers are widely used in NLP, but I wanted to explore:

> Can a Transformer actually learn sequence patterns in continuous signals?

Instead of text, I trained the model on a **sine wave** to test its ability to:
- Understand temporal dependencies
- Predict future values
- Perform autoregressive generation

---

## ⚙️ Architecture

The model is built using:

- 🔹 Projection Layer (for continuous input)
- 🔹 Positional Encoding (sinusoidal)
- 🔹 Transformer Encoder (used as decoder with causal masking)
- 🔹 Linear Output Head

### Key Idea:
A **decoder-only Transformer** can be implemented using:
TransformerEncoder + Causal Mask = Decoder Behavior


---

## 🔧 Components

### 1. Projection Layer
Maps continuous input values to embedding space using an MLP.

### 2. Positional Encoding
Adds sequence order information using sinusoidal encoding.

### 3. Masked Transformer
Causal masking ensures the model only attends to past tokens.

### 4. Output Head
Predicts the next value in the sequence.

---

## 📊 Training Setup

- **Task:** Sequence-to-sequence prediction  
- **Input:** `[x1, x2, ..., xT]`  
- **Target:** `[x2, x3, ..., xT+1]`  
- **Loss:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  

---

## 🔁 Inference (Autoregressive Generation)

The model generates future values step-by-step:

1. Take initial sequence  
2. Predict next value  
3. Append prediction  
4. Repeat  

---

## 📈 Results

- The model successfully learned the sine wave pattern
- It can generate future values beyond the training sequence

### ⚠️ Observation:
- Generated output shows **drift over time**

---

## 💡 Key Insights

- **Low training loss ≠ good generation**
- During inference, the model uses its own predictions → error accumulates
- This is known as **exposure bias**

---

## 🚀 Future Improvements

- Improve long-term stability
- Add time as an explicit feature
- Train on more complex signals (multi-frequency, noisy data)
- Implement KV caching for faster inference

---

## 🛠️ Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib

---

## 📌 Conclusion

This project demonstrates how Transformers can be applied beyond NLP to **continuous sequence modeling**, and highlights the challenges of **autoregressive generation**.

---

## ⭐ Acknowledgment

Built as part of a self-driven exploration into Transformer architectures and sequence learning.
