# What Exactly Is a Trained Model File? Demystifying AI's "Brain Memory Backup"


## Introduction

If you've ever trained a deep learning model, you've definitely encountered files with extensions like `.pth`, `.h5`, or `.onnx`. They might be only tens of megabytes or several gigabytes in size. But have you ever wondered: **what exactly is inside these files?**

<!--more-->

If we compare a well-trained AI to a "master martial artist," then this saved file is the **"backup of the master's brain memory."** It doesn't contain any game graphics, rule code, or human card records—only pure mathematical parameters.

This article will take you through a detailed breakdown of model file components and use "Dinggedan AI" as an example to help you thoroughly understand the essence of modern AI.

## I. What's Inside the File? (Four Core Elements)

When you save a deep learning model, it typically packages the following four components:

### 1. Model Weights and Biases — The Core Soul

These are millions, tens of millions, or even billions of floating-point numbers (Float32 or Float16) that the neural network continuously fine-tunes during "gradient descent."

**In poker AI:** These numerical matrices solidify the AI's "card sense." For instance, a specific combination of weight matrices represents the tactical intuition of "when holding two 4s and the previous player plays a straight, decisively swap cards."

These weights are not randomly generated but are optimal solutions gradually adjusted through thousands of self-play games under the guidance of loss functions. Each number carries the AI's judgment capability for specific situations.

### 2. Network Architecture / Computation Graph

This records how the weights are connected: how many layers there are, how many neurons per layer, and what activation functions are used (such as ReLU).

**Note:** Some frameworks (like PyTorch) only save weights, requiring you to redefine the network structure in code when loading; while others (like TensorFlow SavedModel) bundle both structure and weights together.

The network architecture is like the "skeleton" of the nervous system, determining how information flows between layers. Without this structure, weights would just be a bunch of meaningless numbers.

### 3. Optimizer State — For "Resume Training"

This records the state at training interruption (such as first and second moment estimates in the Adam optimizer).

If power suddenly fails after 3 days of training, with this file, you can continue training from the exact step where it stopped, without starting from scratch.

This is crucial for large model training. Imagine if your AI has already learned from 10 million poker games and the server suddenly crashes—without the optimizer state, you'd have to start over, meaning all previous computational resources would be wasted.

### 4. Metadata and Configuration

This records hyperparameters during training (such as learning rate α, number of epochs trained), vocabulary (for NLP models), or current training steps.

While this information doesn't directly participate in inference, it's very important for reproducing experimental results, continuing training, or debugging the model.

## II. Common File Formats in Mainstream Frameworks

Depending on the deep learning framework you use, the file extensions and formats differ:

### 1. PyTorch (Currently the absolute mainstream in academia and poker AI like DouZero)

- **`.pt` or `.pth`**: The most common extensions. Essentially a compressed Python Dictionary storing the aforementioned weights and tensors using key-value pairs.
  
- **`.safetensors`**: A popular new format in recent years. Safer than `.pth` (won't execute malicious code during loading) and loads extremely fast. Currently strongly recommended by communities like Hugging Face.

**DouZero Example:** After training completion, files like `douzero_checkpoint.pth` are typically generated.

### 2. TensorFlow / Keras

- **`.h5` or `.keras`**: HDF5 format single files containing model structure, weights, and training configuration, very convenient for portability.
  
- **SavedModel (Folder)**: TensorFlow's official standard format. It's a folder containing `saved_model.pb` (computation graph structure) and a `variables/` subfolder (storing specific weight data).

### 3. Cross-Platform Deployment Formats (For deploying AI to mobile phones or web browsers)

- **`.onnx` (Open Neural Network Exchange)**: An open format led by Microsoft. You can convert PyTorch-trained `.pth` models to `.onnx`, then run them efficiently in C++, Java, or even browsers (ONNX.js) without installing the bulky PyTorch library.
  
- **`.engine` / `.trt`**: NVIDIA TensorRT format, specifically optimized for extreme inference acceleration on NVIDIA GPUs.

## III. Concrete Example: "Dinggedan AI"

Suppose you spent a week training a "Dinggedan" DMC model on a server and saved it as `dinggedan_final.pth` (possibly only tens of MB in size).

If you open it with Python code, you'll see a structure like this:

```
python
import torch

# Load model file
checkpoint = torch.load('dinggedan_final.pth')

# View file contents
print(checkpoint.keys())
# Output might look like:
# dict_keys(['model_state_dict', 'optimizer_state_dict', 'epoch', 'loss'])

# View weight parameters
model_weights = checkpoint['model_state_dict']
for key, value in model_weights.items():
    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    
# For example:
# layer1.weight: shape=(128, 64), dtype=torch.float32
# layer1.bias: shape=(128,), dtype=torch.float32
# ...
```

---

> Author: [AZ](https://github.com/0make)  
> URL: https://0make.top/posts/model-weights-explained/  

