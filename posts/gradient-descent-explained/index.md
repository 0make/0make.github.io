# How Does Gradient Descent Actually Adjust Model Parameters? From Micro Operations to Macro Optimization


## Introduction

To understand "how exactly" gradient descent adjusts model parameters, we need to zoom in from the macroscopic "blindfolded mountain descent" to the microscopic "mathematical operation" level.

<!--more-->

Simply put, the essential action of gradient descent parameter adjustment is: **"Based on the 'accountability' result (gradient) of each parameter's contribution to the error, move the parameter slightly in the opposite direction that reduces the error."**

To help you see clearly this "fine-tuning" mechanical process, we break it down into three levels: from micro-level single parameter adjustment, to macro-level network-wide computation, and finally to advanced tuning techniques used in modern large models.

---

## 1. Micro Anatomy: The "Punishment" Process of a Single Weight

Suppose there is only one extremely simple weight parameter $w$ in the neural network (such as the connection line between the "has bomb in hand" feature and "playing win rate"), and the current loss function (error) is $L$.

After forward propagation and backpropagation, we calculate the partial derivative (i.e., gradient) of loss $L$ with respect to weight $w$: $\frac{\partial L}{\partial w}$.

### 1.1 What Does the Gradient Tell Us?

Suppose we calculate $\frac{\partial L}{\partial w} = +2$.

This positive number +2 contains two key pieces of information:

- **Direction (positive sign)**: If $w$ increases, error $L$ will also increase
- **Magnitude (value 2)**: For every 1 unit increase in $w$, error $L$ will increase by 2 units

### 1.2 How Are Parameters Adjusted?

Our ultimate goal is to make error $L$ smaller. Since increasing $w$ leads to increased $L$, we must decrease $w$.

At this point, **Learning Rate (denoted as $\alpha$)** comes into play. Suppose we set $\alpha = 0.1$ (meaning we only dare to fine-tune by 10% each time).

#### Update Formula Execution:

$$w_{new} = w_{old} - \alpha \times \frac{\partial L}{\partial w}$$

Substituting specific numbers (assuming original $w_{old} = 0.5$):

$$w_{new} = 0.5 - (0.1 \times 2) = 0.5 - 0.2 = 0.3$$

**Result**: Weight $w$ is adjusted from 0.5 to 0.3.

**Significance**: Because the model discovered that this weight was "counterproductive" (causing error to increase), it forcefully pulled it in the opposite direction. In the next forward propagation, this feature's impact on the result becomes smaller, and the error decreases accordingly.

> 💡 **Conversely**: If the calculated gradient is -2, it means increasing $w$ can reduce error, so the minus sign in the formula becomes a plus sign, increasing $w$.

---

## 2. Macro Perspective: The "Great Migration" of Millions of Parameters

In real neural networks (such as poker AI or Transformer), there aren't just 1 parameter, but millions or even billions. How does gradient descent adjust them all simultaneously?

### 2.1 Not Loops, But "Matrix Parallelism"

In code (such as PyTorch), we absolutely do not write a `for` loop to update parameters one by one—that would be too slow.

- All weights are organized into huge **Tensors (multi-dimensional matrices)**
- The gradients calculated by backpropagation are also **gradient tensors** with exactly the same shape as the weights
- The parameter update operation is executed at the bottom layer by **GPU performing large-scale matrix subtraction**

### 2.2 The Real Face at Code Level

If you open the source code of deep learning frameworks, the core logic of parameter adjustment (taking SGD as an example) in the underlying C++/CUDA code looks roughly like this:

```python
# Pseudocode: Parallel parameter update on GPU
# params: list of tensors containing all weights and biases
# grads: list of corresponding gradient tensors calculated by backpropagation

for param, grad in zip(params, grads):
    # Core action: parameter = parameter - learning_rate * gradient
    # This step performs parallel subtraction on millions of floating-point numbers simultaneously on GPU
    param.data.sub_(learning_rate * grad.data) 
    
    # Optional: if there is weight decay (regularization), also subtract a portion of the parameter itself
    # param.data.sub_(weight_decay * param.data)
```

This is the **physical truth** of "how exactly adjustment happens": in each training iteration (Step), the GPU multiplies millions of gradient matrices by the learning rate, then forcibly subtracts (or adds) these values from the original weight matrices. The network structure doesn't change, but the "blood (weight values)" flowing inside is completely replaced once.

---

## 3. Modern Advanced: The "Senior Tuners" Behind Large Models

If you only understand the basic formula above (called **SGD, Stochastic Gradient Descent**), you only understand deep learning before 2015.

When training ChatGPT (Transformer LLM) or Stable Diffusion, pure SGD faces two fatal problems:

1. **Easy to get stuck in local minima**: Reaching a small pit halfway up the mountain where gradient is 0, then stopping there
2. **Difficult to unify learning rate**: Some parameters need big fast steps, others need small slow steps; a globally unified learning rate $\alpha$ performs poorly

Therefore, how does modern AI actually tune parameters? The answer is using **advanced optimizers**, the most famous being **Adam (Adaptive Moment Estimation)**.

Adam makes two extremely clever "modifications" to basic gradient descent:

### 3.1 Modification 1: Introducing "Momentum" — Descending Like Pushing a Snowball

**Principle**: Adam not only looks at the current gradient but also records the gradient directions from previous steps.

- If gradients point in the same direction for several consecutive steps, it gives the parameter an "inertia" (acceleration)
- If gradient directions oscillate back and forth, it cancels out this oscillation

**Effect**: Makes parameters "slide" faster on smooth error surfaces and less likely to get stuck in small pits.

### 3.2 Modification 2: Adaptive Learning Rate — Giving Each Parameter Different "Shoes"

**Principle**: Adam records the sum of squared historical gradients for each parameter.

- If a parameter (such as rare word embeddings) is rarely updated, having small squared gradient sum, Adam automatically **increases** its learning rate, letting it learn faster
- If a parameter (such as common words like "the", "is") is updated extremely frequently, having large squared gradient sum, Adam automatically **decreases** its learning rate, preventing it from "over-learning" and destroying existing knowledge

**Effect**: Achieves "personalized" parameter adjustment for each parameter, greatly accelerating convergence.

---

## 4. Summary: The Complete Lifecycle of Parameter Adjustment

Now, we can string together the complete process of "gradient descent adjusting parameters":

### 4.1 Perceiving Error

AI makes predictions and calculates the gap (Loss) between predicted values and true outcomes (such as poker wins/losses).

### 4.2 Backward Accountability

Using calculus chain rule, calculate backwards from the output layer to precisely determine the partial derivatives (gradients) of Loss with respect to every weight in the network. This is equivalent to issuing a "penalty ticket" to each parameter, telling it: "Here's how much you caused the error to increase/decrease this time; the responsibility is yours."

### 4.3 Advanced Decision-Making

Optimizers like Adam receive these "penalty tickets" and, combining historical memory (momentum) and update frequency (adaptive), calculate the actual step size and direction each parameter should truly move.

### 4.4 Physical Execution

GPU activates, performing parallel matrix subtraction between millions of weight matrices and the calculated adjustment amounts.

### 4.5 Rebirth

Old weights are overwritten, new weights are born. The neural network completes a tiny "evolution".

---

> **Core Insight**: After tens of millions of such "perceive → accountability → decision → execute" cycles, those originally randomly generated numbers are gradually sculpted by the invisible chisel of gradient descent into "artificial intelligence" capable of understanding language, winning at poker, and even generating stunningly beautiful images.

---

> Author: [AZ](https://github.com/0make)  
> URL: https://0make.top/posts/gradient-descent-explained/  

