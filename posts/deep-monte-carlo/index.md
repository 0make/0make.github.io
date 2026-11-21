# Deep Monte-Carlo Algorithm (DMC): From Self-Play to Intuitive Decision-Making


## Introduction

In the field of artificial intelligence, game playing has always been one of the most challenging problems. From Go to poker, different types of games impose vastly different requirements on algorithms. Today, we will delve into the **Deep Monte-Carlo Algorithm (DMC)**, an innovative method that has demonstrated remarkable power in imperfect information games.

<!--more-->

To summarize the principle of DMC in the simplest terms: **Instead of relying on logical deduction to "imagine" the next move, it directly sees the "final outcome" through massive simulations, then uses deep neural networks to memorize and summarize "which action in which situation leads to the highest probability of winning."**

## 1. Core Components Breakdown

### 1.1 Monte Carlo (MC) Part: How to Evaluate Actions?

In reinforcement learning, there are traditionally two methods to evaluate the value $Q(S,A)$ of an action $A$ in state $S$:

#### Temporal Difference (TD, such as classic DQN)

Take one step, observe the immediate reward, and add an estimate of the "maximum value of the next step" (imagination). The formula is:

$$\text{Target Value} = \text{Immediate Reward} + \gamma \times \text{Maximum Predicted Value of Next Step}$$

This method relies on estimating future values, which can easily introduce bias.

#### Monte Carlo (MC)

**Refuses imagination**. It requires completing the entire game until a winner is determined. Then it backtracks and uses the final true total return (Return, denoted as $G$) of that game as the value label for every previous action.

**Core characteristic of MC**: It is **unbiased**. Because it uses the actual final outcome that occurred, without introducing any "erroneous estimates of the future."

### 1.2 Deep (Deep Learning) Part: How to Remember Massive Situations?

Traditional Monte Carlo methods use a "lookup table" (Q-Table) to record the value of each state. However, in poker (or Ding Ge Dan), the number of states is astronomical (over $10^{20}$), making tables impossible to store.

**Introduction of Deep Learning**: DMC uses a **Deep Neural Network (DNN)** to replace the lookup table.

**Role of the network**: It acts as a "function approximator." The input is the current situation features (hand cards, historical plays), and the output is the predicted win rate for each legal action.

## 2. Standard Execution Flow of DMC Algorithm

The training of DMC is a closed loop of "**Self-Play → Collect Outcomes → Supervised Learning**." The specific steps are as follows:

### 2.1 Self-Play Sampling

Let the current neural network play against itself (or multiple instances). At each step, the network predicts the values of various actions based on the current situation $S_t$, and selects an action $A_t$ to execute with some randomness, continuing until the game ends.

### 2.2 Calculate Return

The game ends, producing a final result $G$ (for example: win $G=1$, lose $G=-1$).

### 2.3 Build Training Dataset

Extract every step taken in this game to form training samples: $(S_t, A_t, G)$. Note that all steps in this game share the same final outcome $G$ as the label.

### 2.4 Supervised Learning Update

Feed the massively collected samples into the neural network. Calculate the error between the network's predicted value $Q(S_t, A_t)$ and the true outcome $G$ (typically using Mean Squared Error, MSE). Update the neural network's weights through backpropagation, making the network's predictions increasingly closer to the true outcomes.

### 2.5 Iterative Loop

Use the updated, stronger network to restart from step 1, continuously iterating until the network converges into a top-tier expert.

## 3. Why Can DMC "Outclass" Traditional Algorithms in Poker?

This is the soul of understanding DMC's principle. Poker is an **imperfect information game** with enormous hidden information (opponents' hole cards).

### 3.1 Fatal Flaw of Traditional DQN (TD Method): Error Accumulation

In poker, since you cannot see opponents' cards, your estimate of the "next step value" is often wrong. The TD method uses this erroneous estimate to update the current value, causing **errors to accumulate like a snowball**. Eventually, the AI becomes blindly confident, overestimating certain actions that appear promising but are actually fatal.

### 3.2 DMC's (MC Method) Superiority: Law of Large Numbers Eliminates Variance

Although the outcome of a single MC game is greatly affected by "luck" (card distribution), leading to **extremely high variance**, it is **absolutely unbiased**.

DMC's solution is: **Exchange extreme computing power for massive data**. When self-play reaches tens of millions or even hundreds of millions of games, according to the **Law of Large Numbers**, the element of luck is completely smoothed out. The neural network ultimately learns the pure long-term expected win rate, stripped of luck.

## 4. Three Engineering Pillars That Make DMC Feasible

Pure theoretical Monte Carlo methods converge extremely slowly. The reason DMC has succeeded in modern AI (such as [DouZero](https://github.com/kwai/DouZero)) is due to three key engineering optimizations:

### 4.1 Action Embedding

Poker actions are not simple "up/down/left/right," but rather "play three 4s," "play a straight." DMC maps complex card patterns to low-dimensional dense vectors, enabling the neural network to understand that "playing 34567" and "playing 45678" are logically similar, greatly improving generalization ability.

### 4.2 Action Masking

Before the neural network outputs predictions, a hard-coded rule engine forcibly sets the probability of "illegal actions" (such as playing a single card when the opponent plays a bomb) to negative infinity. This not only ensures the AI never violates rules but also allows the network to focus only on a few legal actions, significantly reducing the learning space.

### 4.3 Massive Parallelism

To combat MC's high variance, DMC architectures are typically deployed on hundreds or thousands of CPU cores running self-play simultaneously, generating training data with extremely high throughput, using "**brute force**" to forcefully suppress noise.

## 5. Summary: The Philosophy of DMC

If we compare the traditional DQN algorithm to a **"chess master who loves to deduce moves and emphasizes logical progression"**;

Then Deep Monte-Carlo (DMC) is a **"street gambler who relies on intuition gained from countless battles."**

> **The philosophy of DMC is**: In imperfect information games filled with uncertainty and deception, logical deduction of intermediate processes is often unreliable. Instead of abandoning complex reasoning, directly fit the true final outcomes through massive real-game sessions, letting the deep neural network learn. It doesn't ask you "why this move is good," it only tells you "historical data proves that playing this way leads to the highest win rate."

## 6. Practical Application Prospects

In the "Ding Ge Dan" project, DMC can be applied to the following scenarios:

- **Endgame Deduction**: Utilizing DMC's unbiased nature to precisely evaluate the long-term win rates of various card exchange strategies
- **Agent Training**: Generating high-quality training data through self-play to improve decision quality
- **Strategy Optimization**: Combining action embedding technology to help AI understand the similarity of complex card patterns

Compared to traditional MAPPO/QMIX algorithms, DMC is more suitable for handling highly uncertain scenarios, while multi-agent collaboration algorithms have advantages in team coordination. The combination of both may produce a more powerful hybrid architecture.

---

**References**:
- [DouZero: Mastering Dou Dizhu with Self-Play Deep Reinforcement Learning](https://github.com/kwai/DouZero)
- [Monte Carlo Methods in Reinforcement Learning](https://www.incompleteideas.net/book/thebook-2nd.html)

**Further Reading**:
- [Comparison of Temporal Difference and Monte Carlo Methods in Reinforcement Learning](/posts/)
- [AI Solutions for Imperfect Information Games](/categories/artificial-intelligence/)

---

> Author: [AZ](https://github.com/0make)  
> URL: https://0make.top/posts/deep-monte-carlo/  

