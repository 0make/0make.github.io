# "Prediction" vs "Ground Truth": Where Do AI "Answers" Come From?


## Introduction

> **The Most Intuitive Analogy**: "Prediction" is the answer AI writes down in the exam hall, while "Ground Truth" is the standard answer key held by the teacher.

<!--more-->

The essence of neural network training is to let AI continuously solve problems, then compare its "predictions" with "ground truth labels" to see how wrong it is (calculate Loss), and then adjust parameters (gradient descent).

However, depending on the AI's learning mode (supervised learning vs reinforcement learning), the "sources" of these two things are very different. Let's break it down in detail using the poker AI (Ding Ge Dan) you care about and everyday AI (such as cat recognition).

---

## 1. How Does Ground Truth (Label) Come About?

Ground truth is **"the absolutely correct standard answer"**. Its source depends on what game AI is playing:

### 1.1 In "Supervised Learning" (Such as Cat Recognition, Machine Translation)

**Source**: Human labor annotation or existing data pairs.

- **Cat Recognition**: After human engineers collect 1 million cat images, they must manually label each image. For example, label one image as `Label = 1` (representing cat), and label a dog image as `Label = 0` (representing dog). These labels are hardcoded in the dataset files in advance.

- **Machine Translation**: Chinese-English sentence pairs written by human translators. Input "你好", the standard answer label is "Hello".

### 1.2 In "Reinforcement Learning" (Such as DouZero Playing "Ding Ge Dan", AlphaGo)

**Source**: Final settlement of the game environment or Monte Carlo simulation.

In poker AI, no human will label each move with a "standard answer" (because the situation is too complex for humans to label). Its "ground truth" is "played out" in the virtual environment.

#### Ground Truth ($G$) in DMC (Deep Monte Carlo):

After AI completes a full game of "Ding Ge Dan" at the virtual card table, the system determines win or loss according to the rules.

- If AI wins this round, the environment returns a ground truth: $G = +1$
- If AI loses this round, the environment returns: $G = -1$

This $+1$ or $-1$ is the "ground truth" shared by all steps in this game. It is derived from real execution through physical rules (game engine), without any subjective speculation.

---

## 2. How Does Prediction ($\hat{Y}$) Come About?

Prediction is the result that AI "guesses" after layer-by-layer calculation through its own brain (neural network) based on current input. Its generation process is **Forward Propagation**.

### 2.1 The Production Pipeline of Predictions

1. **Input Features ($X$)**: Convert current state into numbers. For example, encode "Ding Ge Dan" hand cards into a vector containing 300 numbers
2. **Network Calculation**: These 300 numbers enter the neural network, passing through countless "matrix multiplication + activation functions" we discussed earlier, layer by layer
3. **Output Layer ($\hat{Y}$)**: Data reaches the last layer of the network (output layer), spitting out the final prediction

### 2.2 What Does Prediction Look Like? (Depends on Task Type)

#### In Classification Tasks (Such as Image Recognition):

The output layer usually connects with a Softmax activation function. It converts results into probabilities.

**AI's Prediction**: `[Cat probability: 0.85, Dog probability: 0.10, Bird probability: 0.05]`

#### In Regression/Value Estimation Tasks (Such as Poker AI Predicting Win Rate):

The output layer is usually linear output (no activation function, or using Tanh), directly spitting out a specific numerical value.

**AI's Prediction**: The network predicts that playing this card currently has a final win rate of $Q = 0.65$ (i.e., 65% confidence of winning).

---

## 3. When "Prediction" Meets "Ground Truth": Calculating Loss (Error)

Now, AI has submitted its exam paper. The teacher (Loss Function) starts checking answers.

### Scenario A: Cat Recognition (Classification Task)

- **Ground Truth**: This image is clearly a cat, label is `[1, 0, 0]`
- **Prediction**: AI guessed `[0.85, 0.10, 0.05]`
- **Calculate Loss (Cross-Entropy)**: The teacher found that AI guessed correctly, but not confidently enough (didn't give 1.0), so calculated a smaller Loss value (such as 0.15). If AI guessed `[0.1, 0.8, 0.1]` (mistaking cat for dog), the Loss value would explode (such as becoming 2.5)

### Scenario B: Playing "Ding Ge Dan" (Reinforcement Learning / DMC)

- **Ground Truth ($G$)**: This game is over, AI won, the environment gives the true outcome of $+1$
- **Prediction ($Q$)**: At step 10 of the game, AI's neural network predicted its current situation's win rate as $0.4$
- **Calculate Loss (Mean Squared Error MSE)**:

$$Loss = (Prediction - Ground Truth)^2 = (0.4 - 1)^2 = (-0.6)^2 = 0.36$$

The teacher discovered: AI was too pessimistic at step 10 (only predicted 40% win rate), but actually won this round (100%). AI seriously underestimated its hand strength, producing an error of 0.36.

---

## 4. Summary: Their Position in the Training Loop

Now, we can completely map out the "data flow" of the entire deep learning training:

1. **Prepare Answers**: Take out ground truth from the dataset (manually labeled cats, or $+1$ from poker environment settlement)
2. **AI Solves Problems**: Feed input data (image pixels, or current hand cards) to the neural network, go through forward propagation, obtain predictions
3. **Teacher Grades**: Use loss function to compare "predictions" and "ground truth", calculate Loss (error)
4. **Backward Accountability**: Through backpropagation (calculus chain rule), calculate gradients of Loss with respect to each weight
5. **Modify Brain**: Through gradient descent, fine-tune the neural network's weights

---

> **One-Sentence Summary**: Ground truth is objective fact (given by humans, or calculated by rules), it never changes; prediction is AI's subjective guess, it becomes closer and closer to ground truth as network weights are continuously adjusted (gradient descent). When predictions infinitely approach ground truth, we say: this model training has "converged", and AI has graduated.

---

**Series Articles**:
- [Self-Attention Mechanism Intuitively Explained](/posts/self-attention-explained/)
- [How Does Gradient Descent Actually Adjust Model Parameters?](/posts/gradient-descent-explained/)
- [Deep Monte Carlo Algorithm (DMC) Principle Analysis](/posts/deep-monte-carlo/)
- [Deep Neural Network (DNN) Principle Analysis: From Neurons to "Intuitive" Decision-Making](/posts/deep-neural-network/)
- [What Exactly Are the Files Saved After Model Training?](/posts/model-weights-explained/)

**References**:
- [Deep Learning by Ian Goodfellow et al.](https://www.deeplearningbook.org/)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/thebook.html)
- [Understanding the Loss Function in Deep Learning](https://towardsdatascience.com/understanding-loss-functions-in-deep-learning/)

---

> Author: [AZ](https://github.com/0make)  
> URL: https://0make.top/posts/prediction-vs-label/  

