# Deep Neural Network (DNN) Principles: From Neurons to 'Intuitive' Decision-Making


## Introduction

When we talk about artificial intelligence, "Deep Neural Network" (DNN) is undoubtedly one of the most core technologies. But is its principle really that mysterious?

<!--more-->

To summarize in one sentence: **A deep neural network is a "super function fitter" composed of layers of simple mathematical formulas nested together. Its sole purpose is to learn from massive amounts of data to find the extremely complex mapping relationship between input (such as a hand of poker cards) and output (such as the win rate of that hand).**

To thoroughly explain its principles, we break it down into four core modules: **basic units, network structure, learning process (training)**, and **why "depth" is so powerful**.

## 1. Basic Unit: Artificial Neuron

The inspiration for DNN comes from biological brain neurons, but mathematically, it's just a very simple computational unit. A neuron receives multiple inputs and produces an output through two steps of calculation:

### 1.1 Linear Weighted Sum

Multiply each input $x_i$ by a **Weight ($w_i$)**, then add a **Bias ($b$)**.

**Intuitive understanding**: The weight $w$ represents the "importance" of this input. For example, in poker, the weight of the feature "having a bomb in hand" is definitely much larger than "having a single 3 in hand." The bias $b$ acts as a "trigger threshold."

**Formula**:

$$z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

### 1.2 Non-linear Activation

Put the sum result $z$ into an **activation function** (such as ReLU, Sigmoid).

**Intuitive understanding**: If there were only linear summation, no matter how many layers the network stacks, it would ultimately be equivalent to a single layer of linear computation, which is meaningless. The activation function introduces "non-linearity," allowing neurons to decide "whether to be activated" (for example: only when the total card strength score exceeds a certain threshold, output a strong signal).

**Formula**:

$$a = \text{Activation}(z)$$

Common activation functions include:
- **ReLU**: $f(x) = \max(0, x)$, simple and efficient, currently the most commonly used activation function
- **Sigmoid**: $f(x) = \frac{1}{1+e^{-x}}$, compresses output to between 0-1, suitable for probability output
- **Tanh**: $f(x) = \tanh(x)$, compresses output to between -1 and 1

## 2. Network Structure: From Shallow to Deep

Connecting hundreds or thousands of such neurons together forms a neural network. It is typically divided into three layers:

### 2.1 Input Layer

Receives raw data. In poker AI, the input might be a vector encoding:
- "My hand cards"
- "Historical plays"
- "Remaining cards"
- "Opponent behavior patterns" etc.

### 2.2 Hidden Layers

This is the source of "**depth**." The network can have multiple hidden layers (such as 3 layers, 10 layers, or even hundreds of layers). Each layer's neurons receive the output from the previous layer, perform weighting and non-linear transformation, and then pass it to the next layer.

### 2.3 Output Layer

Provides the final result. In the DMC algorithm, the output layer may directly output the "**predicted win rate (Q-value)**" for each legal action in the current situation.

## 3. Core Mechanism: How Does the Neural Network "Learn"?

When a neural network is first initialized, the weights $w$ and biases $b$ inside are randomly generated. At this point, it's an "idiot," and its predictions are pure guesses. Its "learning" process is essentially continuously adjusting these $w$ and $b$ to make predictions more accurate. This process consists of three steps:

### 3.1 Forward Propagation: Make a Prediction

Data enters from the input layer and, like passing through a factory assembly line, undergoes "weighted sum + activation function" calculations layer by layer, finally obtaining a predicted value at the output layer (for example: AI predicts that playing a certain card has a 60% probability of winning).

### 3.2 Loss Calculation: See How Wrong It Is

Compare the AI's predicted value with the true label (in DMC, this is the actual win/loss result after completing the game, such as 1 or -1).

Use a **Loss Function** (such as Mean Squared Error, MSE) to quantify this gap. The larger the gap, the higher the Loss value.

**Mean Squared Error Formula**:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Where $y_i$ is the true value and $\hat{y}_i$ is the predicted value.

### 3.3 Backpropagation & Gradient Descent: Blame Assignment and Correction

This is the most core and ingenious algorithm of DNN (supported by the **chain rule** in calculus).

#### Backpropagation (BP)

Since the final prediction was wrong, who is responsible? The BP algorithm starts from the output layer and works backwards, precisely calculating the "**contribution**" (i.e., gradient) of each weight $w$ and bias $b$ in the network to the final error.

#### Gradient Descent (GD)

After knowing the responsibility of each parameter, the algorithm fine-tunes each $w$ and $b$ along the direction that "reduces error" (the opposite direction of the gradient).

**Intuitive analogy**: It's like descending a mountain blindfolded. You (the error) want to know how to reach the valley bottom (minimum Loss), so you test the surroundings with your feet, find the steepest downward direction (gradient), and take a small step down (Learning Rate). Repeat this process, and eventually you'll reach the valley bottom (optimal parameters).

**Update Formula**:

$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$$

Where $\eta$ is the learning rate and $\frac{\partial L}{\partial w}$ is the gradient of the loss function with respect to the weight.

## 4. Why Is It Called "Deep"? (The Magic of Depth)

Early neural networks had only 1-2 hidden layers (shallow networks), and they could only solve simple problems (such as recognizing handwritten digits). The reason "deep" networks (multi-layer hidden layers) are so powerful is that they possess the ability of **Hierarchical Feature Learning**.

Taking poker AI as an example, when processing data, deep networks automatically form an "**abstraction ladder**":

### First Hidden Layer

May only learn to recognize the most basic patterns, such as:
- "Is there a pair?"
- "Are there consecutive numbers?"

### Second Hidden Layer

Combines features from the first layer to learn to recognize more complex patterns, such as:
- "This is a straight"
- "This is a small bomb"

### Third Layer and Beyond

Further combines to learn to recognize advanced tactical intentions, such as:
- "The opponent might be bluffing"
- "The current situation is suitable for using 4 to exchange cards and intercept"

> **Key difference**: Shallow networks require human engineers to manually design these features (Feature Engineering), while deep networks can "**automatically learn**" these features directly from raw data.

## 5. Summary: The Role of DNN in DMC Algorithm

Returning to the poker AI (DMC algorithm) we discussed earlier:

- **Monte Carlo (MC)** acts as the "**grading teacher**," providing the true final outcome (standard answer $G$) for each situational action through massive simulations.

- **Deep Neural Network (DNN)** acts as the "**student doing practice problems**." It guesses answers through forward propagation and corrects the connections in its brain (weights $w$) by comparing with standard answers through backpropagation.

After tens of millions of rounds of "practice" (self-play training), the internal weights of this deep neural network are adjusted to an excellent state. At this point, when you give it a new poker situation (input) in actual combat, it doesn't need to perform time-consuming Monte Carlo simulations. It only needs to perform a very fast forward propagation to instantly output the action prediction closest to the true win rate based on "**intuition**."

**This is the "chess sense" or "card sense" that deep neural networks give to AI.**

## 6. Practical Application: DNN Deployment in the "Ding Ge Dan" Project

In the "Ding Ge Dan" AI dual-engine architecture, DNN plays a core role:

### 6.1 Feature Engineering Reconstruction

For special rules such as "3 cards as bomb, 4 cards as hun," redesign the feature encoding method of the input layer to ensure the network can understand these unique card pattern logics.

### 6.2 Action Embedding Optimization

Map complex card patterns (such as "play three 4s with a pair") to low-dimensional dense vectors, enabling DNN to capture semantic similarities between different card patterns.

### 6.3 Distributed Training Acceleration

Utilize large-scale parallel computing to simultaneously conduct self-play and data collection on hundreds of CPU cores, significantly shortening the training cycle.

### 6.4 Inference Speed Optimization

After training is complete, DNN's forward propagation can be completed in milliseconds, meeting the performance requirements of real-time battles.

---

**Further Reading**:
- [Deep Monte-Carlo Algorithm (DMC) Principles](/posts/deep-monte-carlo/)
- [Comparison of Temporal Difference and Monte Carlo Methods in Reinforcement Learning](/categories/ai/)
- [Guide to Choosing Neural Network Activation Functions](/tags/deep-learning/)

**References**:
- [Deep Learning by Ian Goodfellow et al.](https://www.deeplearningbook.org/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

---

> Author: [AZ](https://github.com/0make)  
> URL: https://0make.top/resources/ai/deep-neural-network/  

