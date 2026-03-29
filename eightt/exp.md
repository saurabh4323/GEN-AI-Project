# 🎨 GAN for Image Generation (CIFAR-10)

This project implements a **Generative Adversarial Network (GAN)** using TensorFlow to generate realistic images from random noise.

It also demonstrates **latent space interpolation**, showing how GANs learn smooth transitions between generated images.

---

## 📌 Overview

The project uses:

* 🧠 **Generator** → creates fake images
* 🕵️ **Discriminator** → distinguishes real vs fake images

Both networks compete against each other, improving over time.

---

## 📂 Dataset

* **CIFAR-10 Dataset**
* Contains 60,000 images of size **32×32×3**
* Classes include: airplanes, cars, birds, cats, etc.

### 🔄 Preprocessing

* Converted to float32
* Normalized to range **[-1, 1]**

---

## 🧠 Model Architecture

### 🎨 Generator

Transforms random noise into images.

**Input:** 100-dim latent vector
**Output:** 32×32×3 image

Layers:

* Dense → reshape to (8×8×256)
* Conv2DTranspose (upsampling)
* BatchNorm + LeakyReLU
* Final activation: **tanh**

---

### 🕵️ Discriminator

Classifies images as real or fake.

**Input:** 32×32×3 image
**Output:** Real/Fake score

Layers:

* Conv2D (downsampling)
* LeakyReLU + Dropout
* Flatten + Dense(1)

---

## ⚙️ Training Setup

* Loss: Binary Crossentropy (from logits)
* Optimizer: Adam (lr = 1e-4)
* Batch Size: 128
* Epochs: 20

---

## 🔁 Training Process

For each batch:

1. Generate fake images from noise
2. Discriminator:

   * Learns to classify real vs fake
3. Generator:

   * Learns to fool discriminator
4. Backpropagation updates both networks

---

## 🎨 Results

### 🖼️ Generated Images

After training, the generator produces **artistic images** from noise.

---

## 🌈 Latent Space Interpolation

The project visualizes interpolation between two random points in latent space:

* Smooth transition between generated images
* Demonstrates meaningful feature learning

Example:

```
z1 → z2
⬇
smooth image transformation
```

---

## 🔍 Key Concepts

* **GAN (Generative Adversarial Network)**
* **Latent Space Representation**
* **Adversarial Training**
* **Image Synthesis**

---

## ⚠️ Limitations

* Low epochs → images may be blurry
* No class conditioning
* Basic GAN (not DCGAN/StyleGAN level)
* No evaluation metrics (FID, IS)

---

## 🚀 Future Improvements

* Increase epochs (100+)
* Use DCGAN architecture improvements
* Add Conditional GAN (cGAN)
* Implement StyleGAN / Progressive GAN
* Add FID score for evaluation

---

## 🧑‍💻 Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib

---

## ✅ Conclusion

This project demonstrates how GANs:

* Learn to generate realistic images
* Capture complex data distributions
* Enable smooth interpolation in latent space

---

## 📎 Notes

* Suitable for learning GAN fundamentals
* Great starting point before advanced architectures

---
