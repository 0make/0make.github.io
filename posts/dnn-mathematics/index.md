# The Mathematical Essence of Deep Neural Networks: From Composite Functions to Gradient Descent


## Introduction

When we marvel at the amazing capabilities of artificial intelligence, we rarely think about the mathematical foundation behind it. In fact, **the essence of Deep Neural Networks (DNN), mathematically, is an extremely complex composite function**. It fits the mapping relationship between input data and output results through nested layers of simple linear transformations and non-linear activation functions.

<!--more-->

To clearly demonstrate this process, we take the most basic **Multi-Layer Perceptron (MLP, i.e., fully connected feedforward neural network)** as an example, and deconstruct its core mathematical formulas according to the training loop of "**Forward Propagation → Calculate Loss → Backpropagation and Update**."

## 1. Forward Propagation: The Flow of Data

Forward propagation is the process by which the neural network "makes predictions." Data enters from the input layer, calculates layer by layer, and finally reaches the output layer.

### 1.1 Calculation of a Single Neuron

For the $j$-th neuron in layer $l$, it receives the outputs $a^{(l-1)}$ of all neurons in the previous layer (layer $l-1$) as input. Its calculation is divided into two steps:

#### Step 1: Linear Weighted Sum

$$z_j^{(l)} = \sum_i w_{ji}^{(l)} a_i^{(l-1)} + b_j^{(l)}$$

Where:
- $w_{ji}^{(l)}$: The weight between the $j$-th neuron in layer $l$ and the $i$-th neuron in layer $l-1$
- $b_j^{(l)}$: The bias of the $j$-th neuron in layer $l$
- $z_j^{(l)}$: The net input before activation (Linear combination)

#### Step 2: Non-linear Activation

$$a_j^{(l)} = f(z_j^{(l)})$$

Where:
- $f$: Activation function (such as ReLU: $f(z) = \max(0, z)$, or Sigmoid: $f(z) = \frac{1}{1+e^{-z}}$)
- $a_j^{(l)}$: The final output of this neuron, which will serve as input to the next layer

### 1.2 Matrix Form (Actual Calculation Method in Modern Deep Learning Frameworks)

In actual programming (such as PyTorch/TensorFlow), to improve computational efficiency, the calculation of one layer is converted into matrix multiplication. The overall calculation of layer $l$ is as follows:

$$Z^{(l)} = W^{(l)} A^{(l-1)} + B^{(l)}$$

$$A^{(l)} = f(Z^{(l)})$$

Where:
- $A^{(l-1)}$: Output matrix of the previous layer (dimension: $n_{l-1} \times m$, where $m$ is batch size)
- $W^{(l)}$: Weight matrix of layer $l$ (dimension: $n_l \times n_{l-1}$)
- $B^{(l)}$: Bias matrix (dimension: $n_l \times 1$, broadcast to $m$ columns)
- $Z^{(l)}$ and $A^{(l)}$: Net input and activated output matrices respectively

**Boundary Conditions**:
- For the input layer: $A^{(0)} = X$ (i.e., raw input data, such as feature vectors of poker cards)
- For the output layer: $A^{(L)} = \hat{Y}$ (i.e., the final prediction value of the neural network, such as predicted win rate)

## 2. Loss Function: Measuring "How Wrong It Is"

After forward propagation produces the predicted value $\hat{Y}$, it needs to be compared with the true label $Y$ to calculate the error.

### 2.1 Mean Squared Error (MSE) - For Regression Tasks

In reinforcement learning (such as DouZero's DMC algorithm), the network outputs continuous Q values, and MSE is typically used:

$$L(\hat{Y}, Y) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

Where:
- $m$: Number of samples in the batch
- $\hat{y}^{(i)}$: Predicted value of the $i$-th sample
- $y^{(i)}$: True value of the $i$-th sample (such as the true outcome $G$ obtained from Monte Carlo simulation)

> **Note**: The $\frac{1}{2}$ in front is to eliminate the coefficient 2 brought by the square when taking derivatives, making the formula more concise.

### 2.2 Cross-Entropy Loss - For Classification Tasks

If the network outputs probability distributions of various actions (such as AlphaGo's policy network), cross-entropy is typically used:

$$L(\hat{Y}, Y) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{C} y_c^{(i)} \log(\hat{y}_c^{(i)})$$

Where:
- $C$: Total number of categories (such as the number of legal action types in poker)
- $y_c^{(i)}$: One-hot encoding of the true label (1 if it's the correct action, 0 otherwise)
- $\hat{y}_c^{(i)}$: Probability of the $c$-th action predicted by the network

## 3. Backpropagation and Gradient Descent: How the Network Learns

This is the most core mathematical magic of DNN. The goal is to find the partial derivatives (gradients) of the loss function $L$ with respect to every weight $W$ and bias $B$ in the network, then update parameters along the opposite direction of the gradient to minimize the loss $L$.

### 3.1 Chain Rule: Calculating Gradients

According to the chain rule in calculus, the partial derivative of loss $L$ with respect to the weight $W^{(l)}$ of layer $l$ can be decomposed into local partial derivatives multiplied continuously from the output layer all the way to layer $l$:

$$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial A^{(L)}} \cdot \frac{\partial A^{(L)}}{\partial Z^{(L)}} \cdot \frac{\partial Z^{(L)}}{\partial A^{(L-1)}} \cdots \frac{\partial A^{(l)}}{\partial Z^{(l)}} \cdot \frac{\partial Z^{(l)}}{\partial W^{(l)}}$$

This formula looks long, but in code implementation, it is efficiently calculated through **dynamic programming (recursive from back to front)**.

#### Define Error Term (Delta)

$$\delta^{(l)} = \frac{\partial L}{\partial Z^{(l)}}$$

This represents the impact of changes in the net input of layer $l$ on the final loss.

Then the gradient of weights can be expressed extremely concisely as:

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (A^{(l-1)})^T$$

$$\frac{\partial L}{\partial B^{(l)}} = \delta^{(l)}$$

### 3.2 Gradient Descent Parameter Update

After calculating the gradients, use optimization algorithms (such as the basic SGD, or more advanced Adam) to update weights and biases:

$$W^{(l)} \leftarrow W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}$$

$$B^{(l)} \leftarrow B^{(l)} - \alpha \frac{\partial L}{\partial B^{(l)}}$$

Where:
- $\alpha$: **Learning Rate**. It determines the step size of each update. Too large may overshoot the minimum point (diverge), too small converges extremely slowly
- $\leftarrow$: Indicates replacing the old value with the newly calculated value

## 4. Summary: The Mathematical Closed Loop of DNN Training

Connecting the above formulas together, the complete mathematical flow of training a batch of data in a deep neural network is as follows:

1. **Input**: Given data $X$ and true labels $Y$
2. **Forward**: Calculate layer by layer through $A^{(l)} = f(W^{(l)} A^{(l-1)} + B^{(l)})$ to obtain predicted value $\hat{Y} = A^{(L)}$
3. **Calculate Loss**: Calculate the gap between prediction and truth through $L(\hat{Y}, Y)$
4. **Backward**: Use the chain rule to calculate $\frac{\partial L}{\partial W^{(l)}}$ and $\frac{\partial L}{\partial B^{(l)}}$ layer by layer from output layer to input layer
5. **Update**: Update all parameters through $W \leftarrow W - \alpha \nabla_W L$
6. **Loop**: Repeat the above process tens of thousands of times (Epochs) until Loss converges to a minimum

### Mapping in Poker AI (such as DouZero)

- $X$ is the current hand cards and play history features
- $\hat{Y}$ is the Q value (win rate) predicted by the network for each action
- $Y$ is the true win/loss result $G$ obtained after completing a game through Monte Carlo simulation

Through this rigorous set of calculus formulas, the neural network continuously fine-tunes millions of internal weight parameters during tens of millions of self-play games, ultimately solidifying "**poker intuition**" into these matrix numbers.

## 5. Practical Application: DNN Mathematical Principles in the "Ding Ge Dan" Project

In the "Ding Ge Dan" AI dual-engine architecture, DNN's mathematical mechanisms play a key role:

### 5.1 Construction of Feature Vectors

For the special rules of "3 cards as bomb, 4 cards as hun," design specialized input vectors $X$ to ensure the mathematical model can accurately capture these unique card pattern logics.

### 5.2 Selection of Loss Function

Since the DMC algorithm outputs continuous Q values (win rate predictions), Mean Squared Error (MSE) is used as the loss function to directly optimize the gap between predicted values and Monte Carlo true outcomes.

### 5.3 Learning Rate Tuning Strategy

Use larger learning rates in the early training stage for fast convergence, gradually reduce learning rates in later stages for fine-tuning, ensuring the model can both learn quickly and converge stably.

### 5.4 Batch Normalization

Introduce batch normalization technology in deep networks to solve the vanishing gradient problem and accelerate training convergence:

$$\hat{x}^{(k)} = \frac{x^{(k)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y^{(k)} = \gamma \hat{x}^{(k)} + \beta$$

### 5.5 Regularization to Prevent Overfitting

Prevent the model from overfitting training data through L2 regularization (weight decay):

$$L_{total} = L + \lambda \sum ||W||^2$$

Where $\lambda$ is the regularization coefficient.

## 6. Further Reading and Advanced Topics

If you are interested in the mathematical principles of DNN, you can further explore the following topics:

- **Advanced Optimization Algorithms**: Mathematical derivation of adaptive learning rate algorithms like Adam and RMSprop
- **Convolutional Neural Networks (CNN)**: How to extend matrix operations to image processing
- **Recurrent Neural Networks (RNN)**: Mathematical mechanisms for processing sequential data
- **Attention Mechanism**: Core mathematical principles of Transformer architecture
- **Generative Adversarial Networks (GAN)**: Combination of game theory and calculus

---

**Series Articles**:
- [Deep Monte-Carlo Algorithm (DMC) Principles](/posts/deep-monte-carlo/)
- [Deep Neural Network (DNN) Principles: From Neurons to 'Intuitive' Decision-Making](/posts/deep-neural-network/)

**References**:
- [Deep Learning by Ian Goodfellow et al.](https://www.deeplearningbook.org/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/)

---

> Author: [AZ](https://github.com/0make)  
> URL: https://0make.top/posts/dnn-mathematics/  

