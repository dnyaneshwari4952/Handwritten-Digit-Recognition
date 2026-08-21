# MNIST Handwritten Digit Dataset

## Overview
The **MNIST (Modified National Institute of Standards and Technology)** dataset is a foundational benchmark in machine learning and computer vision.

- **Total Samples**: 70,000 grayscale images
  - **Training Set**: 60,000 samples
  - **Testing Set**: 10,000 samples
- **Image Resolution**: 28 x 28 pixels (784 features per image when flattened)
- **Channels**: 1 (Grayscale)
- **Target Classes**: 10 distinct digits (`0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`)
- **Pixel Values**: Integer range `[0, 255]`, where `0` represents pure black background and `255` represents pure white digit stroke.

---

## Dataset Loading & Preprocessing
In this repository, the dataset is loaded programmatically via `src/data_loader.py` using `keras.datasets.mnist`.

### Preprocessing Steps:
1. **Validation**: Check for non-empty arrays, 28x28 shapes, label range [0, 9], and pixel value ranges [0, 255].
2. **Normalization**: Pixel values are scaled linearly from `[0, 255]` to `[0.0, 1.0]` as `float32` tensors:
   $$\hat{x} = \frac{x}{255.0}$$
3. **Reshaping**: Expanded to rank-4 tensor format `(Batch, Height, Width, Channels)` = `(N, 28, 28, 1)`.
4. **Validation Split**: 10% stratified/deterministic validation partition (54,000 train / 6,000 validation).
