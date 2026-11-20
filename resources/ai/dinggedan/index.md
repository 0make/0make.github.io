# Ding Ge Dan AI Algorithm Design: An Innovative Approach Combining Combinatorial Optimization and Multi-Agent Reinforcement Learning


# Ding Ge Dan AI Algorithm Design: An Innovative Approach Combining Combinatorial Optimization and Multi-Agent Reinforcement Learning

## Introduction

"Ding Ge Dan" is a highly distinctive local poker game popular in the Fanshi region of Shanxi Province. Although it also involves 4 players, two decks (108 cards), and 2v2 competition, its unique "card exchange mechanism" and "counter-intuitive card rankings" make it fundamentally different from games like Guandan and Dou Dizhu.

This article provides a detailed introduction to the AI algorithm solution specifically designed for "Ding Ge Dan", which integrates various advanced technologies including combinatorial optimization, multi-agent reinforcement learning, and Monte Carlo Tree Search.

<!--more-->

## Game Rules Overview

### Basic Configuration

- **Number of Players**: 4 players, forming two teams (partners sit opposite each other)
- **Cards Used**: Two standard decks of playing cards, totaling 108 cards
- **Dealing**: Each player initially receives 27 cards
- **Game Objective**: As a team, be the first to play all cards from your hand

### Core Feature: Card Exchange Mechanism

This is the most unique rule that distinguishes "Ding Ge Dan" from all other poker games. The number "4" serves as a "special exchange token" in the game.

**Trigger Condition**: When a player has 2 or more cards with the number 4, they can initiate a card exchange.

**Exchange Target**: Can only exchange for the card combination just played by the previous player (and cannot be a teammate).

**Exchange Formula**: Using N cards of 4, you can exchange for N+1 cards of the same type (Jokers cannot be exchanged).

**Specific Examples**:

- **Single 4**: Cannot be exchanged
- **Two 4s (44)**: Can exchange for any 3-card combination played by the previous player (e.g., three Kings). After successful exchange, you play "44" and gain the right to play
- **Three 4s (444)**: Can exchange for any 4-card combination played by the previous player (e.g., four Aces). After successful exchange, you play "444" and gain the right to play

**Tactical Significance**: Allows players to optimize their hands by converting scattered, unplayable cards into powerful bombs or key combinations through "4"s, achieving "small bets for big gains" and "interceptions".

### Card Combinations and Ranking Rules

"Ding Ge Dan" has many counter-intuitive settings for card combination rankings that require special attention:

**Basic Ranking Order**: Big Joker > Small Joker > 3 > 2 > A > K > Q > J > 10 > 9 > 8 > 7 > 6 > 5 > 4

> **Note**: 3 is the largest numbered card!

**Playing Restrictions**: No accompanying cards allowed (i.e., no rules like "three with one" or "four with two"), must play pure combinations only.

**Standard Combinations**:

- Single cards, pairs
- Straights: Must be 3 or more consecutive cards to play
- Triple Pairs: Must be consecutive pairs (e.g., 33-44-55) to play

**Bomb Hierarchy System (Core Suppression Relationship)**:

1. **Triple Bomb**: 3 cards of the same number. Classified as a "small bomb", cannot suppress triple pairs, but can suppress smaller singles, pairs, etc.
2. **Quadruple Bomb (Hun)**: 4 cards of the same number. Greater than triple bombs, can suppress Jokers and consecutive pairs, as well as singles, pairs, triples, straights of any length, or airplanes
3. **Joker Bomb**: Divided into double Joker bombs, triple Joker bombs, and quadruple Joker bombs. Internal ranking: Double Big Jokers > Mixed Jokers (one big, one small) > Double Small Jokers
4. **Consecutive Bombs (Triple Bomb Chain)**: 3 groups of consecutive bomb combinations (e.g., 333-444-555 or 4444-5555-6666). This is the top-tier combination in the game, capable of suppressing ordinary bombs and double Joker bombs

> **Important Note**: This is why "quadruple Joker bombs are not the strongest"!

### Special Rules and Win/Loss Determination

**Wind Catching**: When a player plays their last combination and no one can beat it (or chooses not to), the playing right transfers directly to their partner, who "catches the wind" and continues playing.

**Winning Algorithm** (assuming AB is one team, CD is the other):

- **Double Capture (Catch Both)**: A finishes 1st, B finishes 2nd. Team AB achieves complete victory, earning 2x base points
- **Single Capture (Catch One)**: A finishes 1st, C (opponent) finishes 2nd, B finishes 3rd. Team AB wins, earning 1x base points
- **Draw**: A finishes 1st, C (opponent) finishes 2nd, D (opponent) finishes 3rd. Both teams tie, earning 0x base points
- **Escape Penalty**: If a player quits mid-game, they lose 12x base points

## AI Algorithm Architecture Design

### 1. Pre-Core: Heuristic Card Exchange Optimization Algorithm

This is the first step that distinguishes "Ding Ge Dan" AI from all other poker AIs. Before each play, the AI must decide "whether to exchange cards" and "how to exchange".

#### Algorithm Selection

Adopts dynamic programming or integer linear programming combined with greedy strategies. These algorithms can find optimal solutions in complex choice spaces, particularly suitable for handling the allocation of special resources like "4"s in "Ding Ge Dan".

#### Core Function

Treats the "4"s in hand as a "universal resource". The AI needs to calculate in real-time: whether using these "4"s to complete scattered cards into bombs, or retaining them as control cards for future use yields greater benefit? This is a variant of the classic knapsack problem, requiring local optimal solutions within milliseconds before feeding the optimized hand into the decision network.

The key to this optimizer lies in balancing short-term benefits and long-term strategy. Sometimes immediate card exchange can gain advantage in the current round, but retaining "4"s might play a more significant role in critical moments later. The algorithm needs to comprehensively consider multiple factors including current hand structure, opponents' possible combinations, teammates' status, and more, to make globally optimal decisions.

#### Working Principle

The optimizer traverses all possible exchange scenarios, evaluating how much each scenario improves the overall strength of the hand. It considers whether exchanges can form stronger bombs, eliminate weaknesses in the hand, affect subsequent playing flexibility, and other factors. Through dynamic programming methods, it avoids redundant calculations, significantly improving solution efficiency.

### 2. Decision Brain: Multi-Agent Reinforcement Learning

Since this is a 2v2 game, the AI must learn to cooperate with teammates (such as blocking, feeding cards, helping teammates finish). This is not a simple single-player decision problem, but a complex system requiring coordination among multiple agents.

#### Algorithm Selection

Adopts MAPPO (Multi-Agent Proximal Policy Optimization) or QMIX algorithms, both utilizing CTDE (Centralized Training with Decentralized Execution) architecture. This architecture allows learning with global information during training, while each agent only needs to make decisions based on local information during execution, making it very suitable for incomplete information game scenarios like "Ding Ge Dan".

#### State Input Design

In addition to regular hand cards and play history, the state vector must separately encode the following key information:

- Number of "4"s held by current player and teammate
- Exchange potential value (calculated based on hand structure)
- Remaining card distribution
- Play history record
- Current round's playing requirements

The design of these features directly affects the AI's understanding of the game situation. Particularly the quantity of "4"s and exchange potential, which are unique key information in "Ding Ge Dan", must be fully reflected in the state representation.

#### Action Space Design

The action space not only includes "what cards to play", but must also include special actions for "executing card exchanges". This means the AI's decision dimensions are more complex than traditional poker games. Through millions of self-play games, the AI automatically learns human-like "card language", such as when a teammate has few cards remaining, deliberately not exchanging and playing small singles to signal the teammate to take control.

Learning this implicit communication capability is a highlight of multi-agent systems. The AI doesn't need explicit communication protocols, but conveys information through playing patterns, which is very similar to how expert human players think.

#### Reward Function Design

The reward function design needs to consider multiple dimensions:

- **Base Score**: Positive reward for successfully playing cards
- **Exchange Reward**: Additional reward for successful exchanges that gain advantage
- **Team Collaboration Reward**: Reward when plays help teammates
- **Penalty Mechanism**: Negative reward for incorrect exchanges or wasting "4"s
- **Victory Reward**: Large reward for team victory
- **Defeat Penalty**: Penalty for team defeat

Careful design of the reward function is key to successful reinforcement learning. It needs to accurately reflect the true objectives of the game while guiding the AI to learn correct strategies. Particularly in "Ding Ge Dan", balancing individual interests and team interests is an important design challenge.

### 3. Endgame Deduction: Monte Carlo Tree Search with Exchange Simulation

In the late game (when a side has fewer than 10 cards), neural network intuition may not be precise enough, requiring search algorithm intervention. At this point, traditional empirical rules are no longer reliable, and optimal strategies need to be found through precise calculation.

#### Algorithm Selection

Adopts Information Set Monte Carlo Tree Search (ISMCTS). This is a search algorithm specifically designed for incomplete information games, capable of handling situations where opponents' hands are unknown.

#### Special Improvements

Building on traditional "deterministic sampling" (randomly guessing opponents' hands), must add "opponent exchange simulation". This is a unique challenge in "Ding Ge Dan". During deduction, the AI needs to assume: "If the opponent has 2 '4's, they will very likely convert that scattered pair into a bomb". Through parallel universe simulations incorporating exchange logic, the AI can more accurately calculate endgame win rates, avoiding being counter-attacked by opponents' sudden "exchange-to-bomb" moves.

This improvement enables the search algorithm to better handle the dynamic nature of "Ding Ge Dan". Traditional poker AIs only need to consider cards opponents might hold, while "Ding Ge Dan" AI also needs to consider new combinations opponents might form through exchanges, greatly increasing search space complexity.

#### Workflow

The search process consists of several phases: First, the selection phase, choosing the most promising branches from the root node down the tree; then the expansion phase, adding new child nodes at leaf nodes; followed by the simulation phase, conducting random simulations from new nodes until game end; finally, the backpropagation phase, propagating simulation results upward to update node statistics.

In each simulation, possible opponent hands are randomly sampled, and possible opponent exchange behaviors are simulated. Through extensive simulations, the algorithm builds estimates of win rates for various situations, thereby selecting optimal action plans.

### 4. Customized Feature Engineering

The neural network's "common sense" needs to be redefined to adapt to "Ding Ge Dan"'s counter-intuitive rules. Traditional poker AI feature engineering methods are not entirely applicable here and require targeted adjustments.

#### Card Power Reassessment

In traditional poker, 4 Jokers are invincible, but in "Ding Ge Dan", quadruple Joker bombs are not the strongest. Feature engineering must adjust weights to help the neural network deeply understand the suppression relationship of "3 cards as bomb, 4 cards as Hun". This requires redesigning the card power evaluation system, not simply following traditional poker experience.

Particularly noteworthy is that "3" is the largest numbered card in "Ding Ge Dan", completely different from most poker games. Feature engineering must explicitly encode this rule, otherwise the AI will experience serious confusion during training.

#### Action Masking Technology

Strictly mask plays that don't comply with "Ding Ge Dan" rules at the output layer (such as exchange operations that don't meet exchange rules), significantly reducing the AI's invalid search space. This not only improves learning efficiency but also ensures AI behavior always complies with game rules.

Action masking implementation needs to consider multiple constraint conditions: whether exchanges are currently possible, exchange legality, whether plays meet current round requirements, etc. By eliminating illegal actions in advance, the AI can focus its limited learning capacity on effective strategy spaces.

#### Key Feature Vector Design

Feature vectors contain information from multiple dimensions:

**Basic Hand Features**: Distribution of cards by rank, reflecting basic hand structure.

**Quantity and Distribution Features of 4s**: Number of "4"s in hand and their combination potential. This is one of the most core features in "Ding Ge Dan".

**Exchange Potential Features**: Based on current hand structure and previous player's play, calculates the number and types of valid exchanges possible.

**Team Status Features**: Teammate's remaining cards, whether teammate is in advantageous position, etc. This is crucial for learning team collaboration.

**Opponent Threat Features**: Opponent's remaining cards, possible combination strength, etc. This helps AI assess current situation danger level.

**Play History Features**: Records of recent rounds' plays, containing rich strategic information.

**Current Situation Features**: Whether it's the first play, what combination needs to be beaten, and other contextual information.

Careful design and combination of these features form the foundation for AI understanding of "Ding Ge Dan". Good feature engineering enables AI to learn effective strategies faster, reducing training time and computational resource requirements.

## Technical Implementation Roadmap

### Phase 1: Basic Framework Construction (Estimated 2-3 weeks)

**Implement Complete Rule Engine for "Ding Ge Dan"**

This is the foundation of the entire project. Need to implement card combination recognition and validation system to ensure accurate judgment of various combination legality; implement exchange rules including trigger condition checking, exchange formula application, state updates after exchanges; implement win/loss determination logic to correctly handle double capture, single capture, draws, and other situations; implement wind catching mechanism to ensure correct transfer of playing rights.

**Build Basic Algorithms for Card Exchange Optimizer**

Implement dynamic programming solver capable of finding optimal exchange solutions within reasonable time; design hand value evaluation function to quantify and evaluate different hand structures; develop exchange decision module integrating optimizer and evaluation function to form complete exchange decision workflow.

**Design State Representation and Action Space**

Complete feature engineering design, determining which features to extract and how to encode them; design action encoding scheme to convert various possible game actions into machine-processable formats; implement action masking mechanism to ensure AI only considers legal action options.

### Phase 2: Reinforcement Learning Training (Estimated 4-6 weeks)

**Set Up MAPPO/QMIX Training Environment**

Encapsulate multi-agent environment providing standard interfaces for reinforcement learning algorithm calls; implement reward functions accurately reflecting game objectives and strategy preferences; establish parallel training framework to improve training efficiency.

**Implement Self-Play Mechanism**

Design Self-Play loop allowing AI to continuously improve through playing against itself; establish experience replay buffer storing historical match data for training; manage opponent pool maintaining opponent diversity to avoid overfitting.

**Train Preliminary Strategy Network**

Build Actor-Critic network architecture, the core component of MAPPO algorithm; perform hyperparameter tuning to find training parameters most suitable for "Ding Ge Dan"; monitor convergence to ensure stable and effective training process.

### Phase 3: Advanced Feature Integration (Estimated 3-4 weeks)

**Integrate ISMCTS Endgame Deduction Module**

Implement exchange simulator capable of simulating opponent exchange behavior during search; merge MCTS with neural networks combining advantages of both; design time control strategy ensuring search completion within limited time.

**Optimize Feature Engineering and Reward Functions**

Conduct feature importance analysis identifying which features have greatest impact on performance; implement reward shaping techniques guiding AI learning through intermediate rewards; adopt curriculum learning strategy gradually transitioning from simple to complex scenarios.

**Conduct Large-Scale Training and Tuning**

Deploy distributed training system to accelerate training process; try model ensemble techniques combining advantages of multiple models; conduct adversarial testing to discover and fix strategy vulnerabilities.

### Phase 4: Performance Optimization and Deployment (Estimated 2-3 weeks)

**Model Compression and Acceleration**

Implement quantization-aware training reducing model size while maintaining precision; apply knowledge distillation technique transferring large model knowledge to small models; export ONNX format for deployment on different platforms.

**Real-Time Decision Optimization**

Optimize inference speed ensuring decision time within acceptable range; design caching strategy avoiding redundant calculations; implement asynchronous decision mechanism improving response speed.

**Deployment Testing and Iteration**

Conduct battle testing with human players collecting real feedback; perform performance benchmark testing quantitatively evaluating system performance; continuously improve based on test results forming virtuous cycle.

## Technical Challenges and Solutions

### Challenge 1: Dynamic Action Space Explosion

**Problem Description**: Exchange mechanism causes exponential growth in action space. Assuming n cards of 4, each 4 can choose to exchange or not, plus considering which combination to exchange for, action space may reach thousands or even tens of thousands of combinations. This makes traditional reinforcement learning methods difficult to apply directly.

**Solutions**:

**Heuristic Pre-screening**: Instead of considering all possible exchange scenarios, only retain valuable options. For example, only consider exchanges that can form bombs or powerful combinations, ignoring those producing weak combinations. This can significantly reduce the number of actions needing evaluation.

**Hierarchical Decision-Making**: Decompose complex decisions into multiple levels. First level decides whether to exchange (a simple binary classification problem); second level decides how many 4s to use if exchanging (a discrete selection problem); third level considers which combination to exchange for (only expanded when necessary). This hierarchical method can decompose one huge action space into multiple smaller sub-problems.

**Action Masking Technology**: Directly mask illegal actions at neural network output layer, reducing无效 exploration and accelerating convergence. This not only improves learning efficiency but also ensures AI behavior always complies with game rules.

### Challenge 2: Asymmetric Team Collaboration

**Problem Description**: Exchange rules stipulate "cannot exchange teammate's cards", meaning AI needs to coordinate actions through implicit communication. How to teach AI to "deliberately not play certain cards" to create exchange opportunities for teammates? This is an advanced collaboration strategy problem.

**Solutions**:

**Encode Teammate's Potential Exchange Capability in State Space**: Based on played card information, infer how many "4"s teammate might have. If estimating teammate might have enough "4"s for exchange, current player should consider leaving exchange opportunities, such as deliberately playing easily exchangeable combinations.

**Transmit Signals Through Playing Patterns**: Deliberately play specific 3-card combinations suggesting teammate can exchange with 2 fours. When teammate has few cards remaining, actively give up playing rights allowing teammate opportunity to clear cards. This implicit communication doesn't require explicit protocols but forms tacit understanding through learning.

**Use Attention Mechanisms to Learn Teammate Behavior Patterns**: Introduce attention mechanisms enabling AI to focus on key information in teammate status. Through learning, AI can discover which teammate features correlate with effective collaboration strategies, thereby better predicting teammate behavior and making corresponding cooperation.

**Introduce Cooperation Rewards**: Give additional rewards when teammate benefits from your plays. This encourages AI to adopt "sacrifice self for greater good" strategies rather than focusing only on personal interests. This reward design is crucial for cultivating genuine team collaboration spirit.

### Challenge 3: Counter-Intuitive Rule Learning

**Problem Description**: Traditional poker experience doesn't apply, such as "3 is the largest single card", "quadruple Joker bombs aren't the strongest". If directly using pre-trained models from general poker AIs, serious negative transfer occurs, meaning previously learned knowledge actually hinders new task learning.

**Solutions**:

**Explicitly Encode Rule Knowledge in Feature Engineering**: Don't use traditional card rank ordering, but use "Ding Ge Dan" specific card power evaluation system. Explicitly tell AI that in this game "3" is the largest numbered card, "4" is the smallest, and has special exchange functionality.

**Use Curriculum Learning Starting from Simple Scenarios**: First phase trains only basic playing without exchange mechanism; second phase adds simple exchange scenarios; third phase enters complex matches under complete rules. This progressive learning approach allows AI to gradually adapt to game complexity.

**Add Rule-Constrained Loss Functions**: During training, if AI violates game rules (such as thinking quadruple Joker bombs are strongest), add extra penalty terms to loss function. This forces AI to quickly correct wrong cognitions.

**Train from Scratch Without Pre-trained Models**: Avoid negative transfer effects, letting AI completely adapt to "Ding Ge Dan"'s unique rules. Although this may require more training time, the final model will be purer and more effective.

## Expected Effects and Evaluation Metrics

### Performance Metrics

**Win Rate Metrics**:

- Against beginner human players: Target win rate over 80%
- Against intermediate human players: Target win rate over 60%
- Against advanced human players: Target win rate over 45%

These targets reflect the level AI should achieve: easily defeating novices, steadily beating intermediate players, and being competitive with experts.

**Decision Efficiency**:

- Average decision time less than 1 second
- Exchange decision time less than 200 milliseconds
- Endgame search depth reaching 10-15 steps

Fast decisions are crucial for good user experience. Players shouldn't feel frustrated waiting for AI to think.

**Exchange Efficiency**:

- Exchange success rate over 70%
- Win rate improvement after exchange averaging over 15%
- "4" usage efficiency: effectively using 2-3 times per game on average

These metrics specifically target "Ding Ge Dan"'s core mechanism—card exchange. Efficient exchange strategy is key to distinguishing ordinary AI from excellent AI.

### Evaluation Methods

**Battle Against Human Players of Different Levels**: Recruit 50 players of varying levels, each playing 100 games against AI, collecting win rates and subjective evaluations. This real-person testing provides the most authentic performance assessment.

**Comparison with Other Benchmark AI Algorithms**: Conduct comparative experiments with DouZero (Dou Dizhu AI), DeepStack (Texas Hold'em AI), random strategy baselines, rule-based baseline AIs, etc. This helps understand "Ding Ge Dan" AI's relative level in the poker AI field.

**A/B Testing Different Algorithm Component Effects**: Separately test with/without exchange optimizer, with/without ISMCTS endgame deduction, impact of different reward functions. Through such ablation experiments, understand each component's contribution, guiding subsequent optimization directions.

### Ablation Experiment Design

Design multiple experiment configurations to evaluate different components' roles:

- **Complete System**: Including all components—exchange optimizer, MAPPO, and ISMCTS
- **Without Exchange Optimization**: Only using MAPPO and ISMCTS, evaluating exchange optimizer's contribution
- **Without Endgame Deduction**: Only using exchange optimizer and MAPPO, evaluating ISMCTS value
- **Rule Baseline Only**: Only using rule-based AI as minimum performance reference

By comparing these configurations' performance, quantify each component's contribution to overall performance, providing basis for resource allocation and optimization priorities.

## Future Expansion Directions

### 1. Online Learning Capability

Implement ability to learn in real-time from human matches. Current system mainly relies on offline training; if online learning were possible, strategies could dynamically adjust based on each opponent's characteristics. Build personalized opponent models recording each player's playing habits and styles, developing targeted counter-strategies.

### 2. Enhanced Explainability

Visualize AI's decision process, allowing players to understand why AI made certain choices. Explain why particular exchange方案 was chosen, based on what considerations. Provide teaching mode AI assistant, not just playing with players but also giving suggestions and guidance, helping players improve their level.

### 3. Multi-Platform Deployment

Develop web-based real-time battle platform allowing more players to conveniently battle AI or watch AI versus AI matches. Develop mobile App allowing players to experience "Ding Ge Dan" anytime, anywhere. Launch offline single-player version satisfying entertainment needs without internet connection.

### 4. Expand to Other Local Poker Games

Migrate this architecture to other poker games with special rules. China has many distinctive local poker games across regions, often with unique rules and gameplay. Establish a universal "local poker AI framework" that can quickly adapt to different game rules, reducing development costs.

## Conclusion

Developing "Ding Ge Dan" AI is an extremely challenging task requiring AI to possess three core capabilities:

**Dynamic Resource Reorganization Capability**: Real-time hand reconstruction through exchange optimizer, solving complex combinatorial optimization problems within milliseconds. This isn't just simple combination recognition, but finding optimal resource allocation schemes in rapidly changing situations.

**Team Tacit Understanding**: Achieving effective collaboration in multi-agent environments, learning implicit communication and tactical coordination. This requires AI to consider not only personal interests but also overall team interests, even making sacrifices when necessary.

**Rule Adaptation Capability**: Quickly learning and adapting to counter-intuitive game rules, breaking free from traditional poker experience constraints. This requires AI to have strong learning capability, summarizing strategy patterns suitable for "Ding Ge Dan" from大量 matches.

By integrating combinatorial optimization, multi-agent reinforcement learning, and Monte Carlo Tree Search technologies, we hope to develop an AI system that truly understands the essence of "Ding Ge Dan". This system will not only achieve excellent performance in games but also provide valuable experience and technical accumulation for developing AIs for other games with similar complex mechanisms.

More importantly, this project demonstrates how to organically combine traditional AI technologies (such as dynamic programming, Monte Carlo search) with modern deep learning methods (such as multi-agent reinforcement learning), providing new ideas and methodologies for solving complex decision-making problems in the real world.

As a local poker game with unique mechanisms, "Ding Ge Dan" provides an excellent testing ground for AI research. Its exchange mechanism introduces the concept of dynamic resource reorganization, its team competition characteristics test multi-agent collaboration capabilities, and its counter-intuitive rules challenge traditional learning paradigms. Successfully developing "Ding Ge Dan" AI will not only bring better gaming experiences to players but also promote AI technology development in fields like game theory, reinforcement learning, and combinatorial optimization.

## References

1. Zha, D., et al. (2021). "DouZero: Mastering Dou Dizhu with Self-Play Deep Reinforcement Learning." *arXiv preprint arXiv:2105.13240*

2. Yu, C., et al. (2022). "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." *NeurIPS 2022*

3. Cowling, P. I., et al. (2012). "Information Set Monte Carlo Tree Search." *IEEE Transactions on Computational Intelligence and AI in Games*

4. Foerster, J., et al. (2018). "Learning to Communicate with Deep Multi-Agent Reinforcement Learning." *ICLR 2018*

5. Brown, N., & Sandholm, T. (2019). "Superhuman AI for multiplayer poker." *Science*

---

**Author's Note**: The algorithm solution introduced in this article is currently in the design phase, with specific implementations to be gradually展开 in subsequent articles. If you're interested in the "Ding Ge Dan" AI project, feel free to leave comments for discussion or participate in open-source contributions!

**Last Updated**: November 20, 2026

---

> Author: [AZ](https://github.com/0make)  
> URL: https://0make.top/resources/ai/dinggedan/  

