# Self-Attention Mechanism Intuitively Explained


## Introduction

> **One-sentence summary**: Self-Attention mechanism allows every element in a sequence (such as each word in a sentence) to "rate" all other elements in the sequence, and then decides who to "focus on" when understanding itself based on the scores.

<!--more-->

Self-Attention is the soul of the Transformer architecture and the key that enables modern Large Language Models (LLMs) to understand context. We can temporarily set aside complex mathematics and first feel it through an intuitive life scenario, then break down its core logic and formulas.

This article will take you from the "social circle" of words, through the perfect analogy of library book searching, and finally into the mathematical essence of Q, K, V, allowing you to thoroughly understand this core technology that changed the AI landscape.

---

## 1. The Most Intuitive Analogy: The "Social Circle" of Words

Suppose we have this sentence:

> "Apple" company released new "Apple" phone.

In traditional Recurrent Neural Networks (RNN), AI reads words one by one from left to right. When it reaches the second "Apple", it may have somewhat forgotten the previous context.

But in **Self-Attention mechanism**, AI sees the **entire sentence at once**. When processing the second "Apple", it sends signals to all words in the sentence for "relevance scoring":

| Word Seen | Score | Reason |
|-----------|-------|--------|
| "company" | +0.8 | Because companies can own Apple |
| "released" | +0.9 | Because the subject of the release action is company/Apple |
| "phone" | +0.95 | Because phone directly modifies Apple, determining that this "Apple" is a tech product, not a fruit |
| "new", "." | +0.1 | Because they are function words with low information content |

**Result**: When updating its internal representation, the second "Apple" absorbs features from "phone", "released", and "company" in large amounts, thereby accurately understanding its meaning here. This is "attention" - allocating limited computational resources to the most relevant words.

This mechanism solves two fatal flaws of traditional RNN:
1. **Long-range dependency problem**: No matter how far apart two words are, they can establish direct connections
2. **Parallel computation advantage**: No need to process sequentially, can compute attention weights for all words simultaneously

---

## 2. The Core Trio: Query (Q), Key (K), Value (V)

The reason Self-Attention can achieve the above process relies on three extremely clever vectors (can be understood as three different "feature extractors"): **Q, K, V**.

We can perfectly metaphor them using **"Library Book Searching"**:

- **Query (Q, query vector)**: Represents **"Who am I now, what am I looking for"**. Like the "borrowing request form" in your hand.
- **Key (K, key vector)**: Represents **"What features do I have, what can I provide"**. Like the "labels/indexes" on each book spine in the library.
- **Value (V, value vector)**: Represents **"My actual content/essence"**. Like the "specific content" after opening the book.

### Workflow

When word A in the sentence wants to update its understanding:

1. Word A takes out its **Q** (need)
2. Word A uses this Q to match and calculate (usually dot product) with **K** (labels) of all words in the sentence (including itself). Higher match means higher score
3. Convert these scores into probability weights (sum to 1) through **Softmax** function
4. Finally, use these weights to perform weighted summation on **V** (actual content) of all words

**Result**: Word A obtains a new "super representation" that integrates the essence of the entire sentence.

### Why Three Vectors?

You might ask: Why not directly use the word's own vector, but separately generate Q, K, V?

The answer lies in **flexibility**:
- The same word may need to play different roles in different contexts
- Q focuses on "what I want to know"
- K focuses on "what information I can provide"
- V focuses on "what my substantive content is"

By linearly transforming the original word vector into three different spaces, the model can learn richer semantic relationships.

---

## 3. The Core Trio: Query (Q), Key (K), Value (V)

The reason Self-Attention can achieve the above process relies on three extremely clever vectors (can be understood as three different "feature extractors"): **Q, K, V**.

We can perfectly metaphor them using **"Library Book Searching"**:

- **Query (Q, query vector)**: Represents **"Who am I now, what am I looking for"**. Like the "borrowing request form" in your hand.
- **Key (K, key vector)**: Represents **"What features do I have, what can I provide"**. Like the "labels/indexes" on each book spine in the library.
- **Value (V, value vector)**: Represents **"My actual content/essence"**. Like the "specific content" after opening the book.

### How Are Q, K, V Generated?

**Q, K, V are not manually set, nor are they fixed. They are dynamically calculated from each word's "initial word embedding" through "matrix multiplication" during the model's training process.**

In other words, Q, K, V are "learned" by the model itself.

To help you thoroughly understand this process, let's break down how this "magic" happens in four steps.

#### Step 1: Starting Point — The Word's "Initial ID Card" (Word Embedding)

When a sentence (such as "Apple released phone") is input to Transformer, the model first converts each word into a fixed-length string of numbers, called Word Embedding.

For example, "Apple" might be represented as a 512-dimensional vector: `[0.2, -0.5, 0.8, ..., 0.1]`.

This initial vector contains the basic meaning of the word in the dictionary, but it is not yet Q, K, or V. We call the matrix containing all word vectors $X$.

#### Step 2: Core Magic — Three Different "Filters" (Weight Matrices)

Inside the Transformer model, there are three learnable weight matrices, which we call $W_Q$, $W_K$, and $W_V$.

When the model first starts training, the numbers in these three matrices are completely random.

As the model trains through massive amounts of text using "gradient descent", the numbers in these three matrices are continuously fine-tuned, eventually learning how to extract the most useful information.

You can think of these three matrices as three different "filters" or "questioners":

- **$W_Q$ (Query filter)**: Specifically used to extract "what information does this word currently need"
- **$W_K$ (Key filter)**: Specifically used to extract "what feature labels can this word provide"
- **$W_V$ (Value filter)**: Specifically used to extract "the actual semantic content of this word"

#### Step 3: Calculation Process — Simple Matrix Multiplication

The process of determining Q, K, V is mathematically extremely simple—just a matrix multiplication:

$$Q = X \cdot W_Q$$

$$K = X \cdot W_K$$

$$V = X \cdot W_V$$

**How exactly is it calculated?**

Suppose "Apple"'s initial word vector $x_{apple}$ is a 512-dimensional vector, and $W_Q$ is a $512 \times 64$ matrix.

When $x_{apple}$ is multiplied by $W_Q$, the 512-dimensional initial information is "compressed" and "reorganized" into a new 64-dimensional vector, which is $q_{apple}$ for "Apple" in this specific context.

Similarly, using the same $x_{apple}$ to multiply with $W_K$ and $W_V$ yields $k_{apple}$ and $v_{apple}$.

> 💡 **Key point**: The same word (same initial vector $X$), because it's multiplied by different weight matrices ($W_Q, W_K, W_V$), splits into three different vectors (Q, K, V), each undertaking different tasks.

#### Step 4: Intuitive Analogy: The Job Seeker's "Triple Identity"

To better understand, let's imagine each word as a job seeker (initial word vector $X$), and $W_Q, W_K, W_V$ as three different evaluation forms set up by the HR department:

1. **Calculating Q (Query)**: HR gives the job seeker a form: "What kind of team collaboration are you currently most eager to find?" After filling it out, the job seeker gets his Q (need).
2. **Calculating K (Key)**: HR gives the job seeker another form: "Please list your top 3 core skill tags." After filling it out, the job seeker gets his K (label).
3. **Calculating V (Value)**: HR gives the job seeker a third form: "Please describe in detail your past actual project experience and capabilities." After filling it out, the job seeker gets his V (substantive content).

After all job seekers (all words in the sentence) complete these three forms, the self-attention mechanism begins to operate:

"Apple" takes its Q (need) and matches/scores against the K (labels) of "released" and "phone". Finding an extremely high match with "phone"'s K, "Apple" then absorbs a large amount of "phone"'s V (substantive content), updating its understanding of the current context.

### Workflow Summary

When word A in the sentence wants to update its understanding:

1. Word A takes out its **Q** (need)
2. Word A uses this Q to match and calculate (usually dot product) with **K** (labels) of all words in the sentence (including itself). Higher match means higher score
3. Convert these scores into probability weights (sum to 1) through **Softmax** function
4. Finally, use these weights to perform weighted summation on **V** (actual content) of all words

**Result**: Word A obtains a new "super representation" that integrates the essence of the entire sentence.

### Why Three Vectors?

You might ask: Why not directly use the word's own vector, but separately generate Q, K, V?

The answer lies in **flexibility**:
- The same word may need to play different roles in different contexts
- Q focuses on "what I want to know"
- K focuses on "what information I can provide"
- V focuses on "what my substantive content is"

By linearly transforming the original word vector into three different spaces, the model can learn richer semantic relationships.

---

## 4. Multi-Head Attention

In actual Transformers, usually not just one set of Q, K, V is used, but multiple sets (called "heads").

### Why Multi-Head?

You might ask: Is one set of $W_Q, W_K, W_V$ enough?

The answer is: No. Because a word often has multiple meanings in a sentence.

This is why Transformer introduces **"Multi-Head Attention"**.

The model doesn't just learn one set of $W_Q, W_K, W_V$, but learns 8 sets, 12 sets, or even more (called 8 "heads").

- **Head 1**'s weight matrix might learn to focus on grammatical relationships (e.g., "Apple"'s Q matches "released"'s K to determine subject-verb relationship)
- **Head 2**'s weight matrix might learn to focus on semantic modification (e.g., "Apple"'s Q matches "phone"'s K to determine it's a tech product)
- **Head 3**'s weight matrix might learn to focus on reference relationships (e.g., the second "it"'s Q matches the first "Apple"'s K)

Imagine:
- **Head 1** might focus on grammatical relationships (subject-verb-object structure)
- **Head 2** might focus on semantic relationships (synonyms, antonyms)
- **Head 3** might focus on reference relationships (which noun does the pronoun refer to)
- **Head 4** might focus on sentiment orientation (positive, negative)

Each head understands the sentence from different perspectives, and finally concatenates the outputs of all heads to form a more comprehensive representation.

### Multi-Head Attention Calculation Process

1. Project Q, K, V into $h$ different subspaces respectively
2. Independently calculate attention in each subspace
3. Concatenate outputs of all heads
4. Then pass through a linear transformation to get the final output

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each head:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Finally, the model concatenates the results calculated by these 8 heads, then undergoes another linear transformation to obtain the word's final, extremely rich contextual representation.

---

## 5. Intuitive Breakdown of the Formula

The core formula of Self-Attention is very elegant and is the soul of the entire Transformer:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Let's break it down into four steps (assuming we're processing a sentence, Q, K, V are all matrices, each row represents a word):

### Step 1: Calculate Relevance Scores ($QK^T$)

$Q$ multiplied by the transpose of $K$ ($K^T$), mathematically means calculating the dot product (inner product) of $Q$ with each $K$.

> **Physical meaning**: The larger the dot product, the more consistent the directions of the two vectors, meaning the "query need" and "feature label" match better, resulting in higher relevance scores.

For example:
- If Q points in the "tech product" direction and K also points in the "tech product" direction, the dot product will be large
- If Q points in the "fruit" direction while K points in the "tech product" direction, the dot product will be small or even negative

### Step 2: Scaling ($\div \sqrt{d_k}$)

$d_k$ is the vector dimension. Dividing by $\sqrt{d_k}$ prevents dot product results from becoming too large.

> **Physical meaning**: If values are too large, the next Softmax function will extreme probabilities (become 1 and 0), causing gradient vanishing and making the model unable to learn. This is an engineering "stabilizer".

**Example**:
- Assume vector dimension $d_k = 64$
- Without scaling, dot products might range from [-100, 100]
- After dividing by $\sqrt{64} = 8$, the range becomes [-12.5, 12.5]
- This makes Softmax output smoother, beneficial for gradient propagation

### Step 3: Normalize Weights ($\text{softmax}$)

Convert all scores obtained in the previous step into a probability distribution summing to 1 through the Softmax function.

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

> **Physical meaning**: Transform raw "matching scores" into "attention weights". For example: 60% attention to "phone", 30% to "released", 10% to other words.

Softmax characteristics:
- Sum of all output values is 1
- Larger input values are amplified, smaller ones are compressed
- Ensures interpretability of weights (probability distribution)

### Step 4: Weighted Summation ($\times V$)

Use the weights calculated in step 3 to multiply with each word's corresponding $V$ (actual content), then sum them all up.

> **Physical meaning**: This is the final "fusion" process. The current word absorbs information from the entire sentence, but the absorption ratio is entirely determined by attention weights.

Mathematical expression:
$$\text{Output}_i = \sum_j \text{weight}_{ij} \cdot V_j$$

Where $\text{weight}_{ij}$ is the attention weight of word $i$ to word $j$.

---

## 6. Multi-Head Attention

In actual Transformers, usually not just one set of Q, K, V is used, but multiple sets (called "heads").

### Why Multi-Head?

Imagine:
- **Head 1** might focus on grammatical relationships (subject-verb-object structure)
- **Head 2** might focus on semantic relationships (synonyms, antonyms)
- **Head 3** might focus on reference relationships (which noun does the pronoun refer to)
- **Head 4** might focus on sentiment orientation (positive, negative)

Each head understands the sentence from different perspectives, and finally concatenates the outputs of all heads to form a more comprehensive representation.

### Multi-Head Attention Calculation Process

1. Project Q, K, V into $h$ different subspaces respectively
2. Independently calculate attention in each subspace
3. Concatenate outputs of all heads
4. Then pass through a linear transformation to get the final output

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each head:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

---

## 7. Why Called "Self" Attention?

In early machine translation models (such as Seq2Seq + Attention), the attention mechanism was cross-sequence: for example, using Q from Chinese sentences to match K from English sentences, called **"Cross-Attention"**.

The brilliance of **Self-Attention** lies in: **Q, K, V all come from the same sequence (the same sentence) itself**.

It doesn't need external target sequences, but allows elements within the sentence to engage in "self-reflection" and "mutual communication". Through this dense internal interaction, the model can directly establish connections between any two words (no matter how far apart they are in the sentence) in just one layer of computation.

### Self-Attention vs Cross-Attention

| Feature | Self-Attention | Cross-Attention |
|---------|----------------|-----------------|
| Q, K, V Source | Same sequence | Different sequences |
| Application Scenario | Inside encoder, inside decoder | Connection between decoder and encoder |
| Function | Understand intra-sequence relationships | Establish connections between source and target sequences |
| Example | Understand the meaning of "Apple" in the sentence | Translate Chinese "苹果" to English "apple" |

---

## 8. Positional Encoding

Self-Attention has an inherent flaw: **it doesn't care about word order**.

Because it sees the "entire sentence at once", "I love you" and "You love me" might look the same to pure Self-Attention (just different weight distributions).

To solve this problem, Transformer introduces **Positional Encoding**, adding position information in the sentence to each word:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:
- $pos$ is the word's position in the sentence
- $i$ is the dimension index
- $d_{model}$ is the model dimension

Thus, even if two words have the same content, as long as their positions differ, their final representations will be different.

---

## 9. Practical Applications: Potential of Self-Attention in "Ding Ge Dan" Project

Although "Ding Ge Dan" currently mainly uses DMC + DNN architecture, Self-Attention mechanism also has application value in certain scenarios:

### 9.1 Modeling Play History

In poker games, play history is sequential data. Using Self-Attention can:
- Capture long-term dependency relationships (such as playing strategies from 10 rounds ago)
- Identify opponent's playing patterns
- Simultaneously focus on multiple key historical nodes

### 9.2 Deep Understanding of Hand Features

Treating hands as a sequence (each card is an element), Self-Attention can:
- Discover combination relationships between cards
- Automatically learn which card types are more important
- Be more flexible than manually designed features

### 9.3 Integration with LLM

As mentioned in our [Transformer LLM vs Poker AI Technology Path Comparison](/posts/transformer-vs-poker-ai/):
- **DMC/DNN** handles precise probability calculation and playing decisions
- **LLM (based on Self-Attention)** handles opponent modeling, tactical analysis, and macro strategy

The two can work synergistically to form a stronger AI system.

---

## 10. Summary

The revolutionary aspects of Self-Attention mechanism:

1. **Global vision**: No longer limited to local windows or sequential processing
2. **Dynamic weights**: Dynamically adjust focus based on context
3. **Parallel computation**: Fully utilize modern GPU parallel capabilities
4. **Interpretability**: Attention weights can be visualized, helping us understand the model's decision-making process

> **Core idea**: Not all information is equally important; learning to "focus" is the key to intelligence.

From the "social circle" of words to the Q/K/V model of libraries, and then to elegant mathematical formulas, Self-Attention mechanism solves the core challenges of sequence modeling in a concise yet powerful way. This is why it became the cornerstone of Transformer architecture and ultimately gave birth to world-changing large language models like GPT and BERT.

---

**Series Articles**:
- [Deep Monte Carlo Algorithm (DMC) Principle Analysis](/posts/deep-monte-carlo/)
- [Deep Neural Network (DNN) Principle Analysis: From Neurons to "Intuitive" Decision-Making](/posts/deep-neural-network/)
- [What Exactly Are the Files Saved After Model Training?](/posts/model-weights-explained/)
- [Transformer LLM vs Poker AI Technology Path Comparison](/posts/transformer-vs-poker-ai/)

**References**:
- [Attention Is All You Need (Transformer Original Paper)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Deep Learning by Ian Goodfellow et al.](https://www.deeplearningbook.org/)

---

> Author: [AZ](https://github.com/0make)  
> URL: https://0make.top/posts/self-attention-explained/  

