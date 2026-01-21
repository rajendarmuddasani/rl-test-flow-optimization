# Reinforcement Learning Interview Guide

Common RL questions with clear, concise answers for technical discussions.

## Table of Contents

1. [Fundamental Concepts](#fundamental-concepts)
2. [Q-Learning Specific](#q-learning-specific)
3. [Comparison Questions](#comparison-questions)
4. [Practical Application](#practical-application)
5. [Advanced Topics](#advanced-topics)
6. [Problem-Solving Questions](#problem-solving-questions)

---

## Fundamental Concepts

### Q1: What is Reinforcement Learning?

**Answer:**

Reinforcement Learning is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative rewards over time through trial and error.

**Key points:**
- Learning from interaction, not labeled examples
- Goal is to maximize cumulative rewards
- Balances exploration (trying new things) and exploitation (using known strategies)

---

### Q2: Explain the difference between supervised learning, unsupervised learning, and reinforcement learning.

**Answer:**

| Aspect | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|--------|-------------------|---------------------|----------------------|
| **Input** | Labeled data (X, y pairs) | Unlabeled data | Environment interactions |
| **Feedback** | Correct answer for each input | None | Rewards/penalties |
| **Goal** | Predict correct output | Find patterns/structure | Maximize cumulative reward |
| **Example** | Image classification | Clustering customers | Game playing, robotics |

---

### Q3: What are the key components of an RL system?

**Answer:**

The five key components are:

1. **Agent**: The learner/decision-maker
2. **Environment**: The world the agent interacts with
3. **State**: Current situation of the environment
4. **Action**: Choices the agent can make
5. **Reward**: Feedback signal indicating action quality

Additionally:
- **Policy (π)**: Strategy for selecting actions
- **Value Function (V or Q)**: Expected cumulative reward

---

### Q4: What is a Markov Decision Process (MDP)?

**Answer:**

An MDP is the mathematical framework for RL problems, defined by:
- **S**: Set of states
- **A**: Set of actions
- **P**: Transition probabilities P(s'|s,a)
- **R**: Reward function R(s,a,s')
- **γ**: Discount factor

**Markov Property**: The future depends only on the current state, not on the history. This means P(s_{t+1} | s_t, a_t) is independent of earlier states.

---

### Q5: What is a policy in RL?

**Answer:**

A **policy** is the agent's strategy for selecting actions. It maps states to actions.

**Types:**
- **Deterministic**: π(s) → a (always same action for same state)
- **Stochastic**: π(a|s) → probability distribution over actions

**Optimal policy π***: The policy that maximizes expected cumulative reward.

---

### Q6: What is the difference between value function and Q-function?

**Answer:**

**Value Function V(s)**:
- Estimates expected cumulative reward starting from state s
- Answers: "How good is this state?"
- Formula: V(s) = E[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t = s]

**Q-Function Q(s, a)**:
- Estimates expected cumulative reward from state s, taking action a
- Answers: "How good is this action in this state?"
- Formula: Q(s, a) = E[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t = s, a_t = a]

**Relationship**: V(s) = max_a Q(s, a)

---

### Q7: What is the discount factor (γ) and why is it important?

**Answer:**

The **discount factor** (0 ≤ γ ≤ 1) determines how much we value future rewards relative to immediate rewards.

**Interpretation:**
- **γ = 0**: Only immediate rewards matter (myopic agent)
- **γ = 1**: All future rewards equally important (far-sighted agent)
- **γ = 0.9-0.99**: Typical values (balance)

**Importance:**
- Ensures convergence (infinite sum becomes finite)
- Reflects uncertainty about future
- Controls planning horizon

---

### Q8: Explain exploration vs exploitation tradeoff.

**Answer:**

**Exploration**: Trying new actions to discover potentially better strategies
- Benefit: May find better solutions
- Cost: Short-term performance suffers

**Exploitation**: Using the best known action to maximize immediate reward
- Benefit: Maximize current performance
- Cost: May miss better strategies

**Tradeoff**: Need to explore enough to find good strategies, but exploit enough to perform well.

**Common solution**: ε-greedy strategy
- With probability ε: explore (random action)
- With probability 1-ε: exploit (best known action)
- Decay ε over time: start exploring, end exploiting

---

## Q-Learning Specific

### Q9: What is Q-Learning?

**Answer:**

Q-Learning is a model-free, off-policy RL algorithm that learns the optimal action-value function Q*(s, a).

**Update Rule:**
```
Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
```

**Key characteristics:**
- **Model-free**: Doesn't require knowledge of environment dynamics
- **Off-policy**: Learns optimal policy while following different policy (e.g., ε-greedy)
- **Temporal Difference**: Updates estimates based on other estimates

---

### Q10: Explain the Q-Learning update rule.

**Answer:**

```
Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
```

**Components:**
- **Q(s, a)**: Current Q-value estimate
- **α**: Learning rate (how much to update)
- **r**: Immediate reward received
- **γ**: Discount factor
- **max Q(s', a')**: Best Q-value in next state (target)
- **[...]**: TD error (difference between target and current)

**Process:**
1. Calculate target: r + γ max Q(s', a')
2. Calculate error: target - current
3. Update: move current toward target by α

---

### Q11: What is the learning rate (α) and how do you choose it?

**Answer:**

The **learning rate** (0 < α ≤ 1) controls how much to update Q-values with new information.

**Effect:**
- **High α (e.g., 0.5)**: Fast learning, volatile, may overshoot
- **Low α (e.g., 0.01)**: Slow learning, stable, better convergence
- **Typical value**: 0.1

**Choosing α:**
- Start with 0.1
- Decrease if learning is unstable
- Increase if learning is too slow
- Can use adaptive learning rates (decrease over time)

---

### Q12: What is temporal difference (TD) learning?

**Answer:**

**Temporal Difference learning** updates value estimates based on other estimates (bootstrapping), rather than waiting for final outcomes.

**TD Error**: δ = r + γV(s') - V(s)
- Difference between estimated value and actual experience

**Advantages:**
- Learn online (don't need to wait for episode end)
- More sample efficient than Monte Carlo
- Can learn in continuing (non-episodic) tasks

**Q-Learning uses TD**: Updates Q(s,a) based on r + γ max Q(s',a')

---

### Q13: What is off-policy learning?

**Answer:**

**Off-policy** learning means the agent learns about one policy (target policy) while following a different policy (behavior policy).

**Q-Learning is off-policy:**
- **Target policy**: Optimal greedy policy (always choose best action)
- **Behavior policy**: ε-greedy (sometimes random, sometimes best)

**Advantage**: Can learn optimal policy while exploring

**Contrast with on-policy (SARSA)**: Learns about the policy it's following

---

## Comparison Questions

### Q14: Q-Learning vs SARSA?

**Answer:**

| Aspect | Q-Learning | SARSA |
|--------|-----------|-------|
| **Type** | Off-policy | On-policy |
| **Update** | Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)] | Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)] |
| **Next action** | Best action (max) | Actual action taken |
| **Behavior** | More aggressive | More conservative |
| **Convergence** | To optimal policy | To followed policy |

**When to use:**
- **Q-Learning**: When exploration is safe, want optimal policy
- **SARSA**: When exploration is risky, want safe policy

---

### Q15: Tabular Q-Learning vs Deep Q-Networks (DQN)?

**Answer:**

| Aspect | Tabular Q-Learning | Deep Q-Networks |
|--------|-------------------|-----------------|
| **Representation** | Q-table (dictionary) | Neural network |
| **State space** | Small, discrete | Large, continuous |
| **Memory** | Stores all Q-values | Stores network weights |
| **Generalization** | None | Generalizes to unseen states |
| **Example** | Grid world, test selection | Atari games, robotics |

**When to use:**
- **Tabular**: Small state/action spaces, want interpretability
- **DQN**: High-dimensional states (images), large state spaces

---

### Q16: Value-based vs Policy-based methods?

**Answer:**

**Value-based (Q-Learning, DQN)**:
- Learn value function Q(s,a) or V(s)
- Derive policy from values (choose action with highest Q-value)
- Works well for discrete actions
- More sample efficient

**Policy-based (REINFORCE, PPO)**:
- Learn policy π(a|s) directly
- No value function needed
- Works for continuous actions
- Can learn stochastic policies

**Actor-Critic**: Combines both (learns value function and policy)

---

## Practical Application

### Q17: How would you apply RL to a real-world problem?

**Answer:**

**Steps:**

1. **Define the problem as MDP:**
   - States: What information is available?
   - Actions: What decisions can be made?
   - Rewards: What is the objective?

2. **Design the environment:**
   - Implement state transitions
   - Define reward function
   - Set episode termination conditions

3. **Choose algorithm:**
   - Discrete actions, small state space → Q-Learning
   - Continuous states/actions → DQN, PPO
   - Safety-critical → SARSA, safe RL

4. **Train and evaluate:**
   - Start with simple baseline (random policy)
   - Train RL agent
   - Compare performance
   - Tune hyperparameters

5. **Deploy and monitor:**
   - Test in simulation first
   - Gradual rollout
   - Continuous monitoring and retraining

---

### Q18: How do you design a good reward function?

**Answer:**

**Principles:**

1. **Align with objective**: Reward what you actually want
2. **Informative**: Provide clear feedback
3. **Balanced**: Not too sparse or too dense
4. **Avoid reward hacking**: Don't create loopholes

**Example (test flow selection):**
- **Good**: +50 for detecting defect, -cost for each test
- **Bad**: +1 for any test (agent will do unnecessary tests)

**Reward shaping**: Add intermediate rewards to guide learning
- Be careful: can change optimal policy
- Use potential-based shaping for guarantees

---

### Q19: What are common challenges in applying RL?

**Answer:**

**Main challenges:**

1. **Sample inefficiency**: Requires many interactions
   - Solution: Experience replay, transfer learning

2. **Sparse rewards**: Rewards are rare
   - Solution: Reward shaping, curriculum learning

3. **Credit assignment**: Which action caused reward?
   - Solution: Discount factor, eligibility traces

4. **Exploration**: Finding good strategies
   - Solution: ε-greedy, UCB, curiosity-driven exploration

5. **Non-stationarity**: Environment/policy changes
   - Solution: Continuous learning, adaptive rates

6. **Safety**: Exploration can be dangerous
   - Solution: Safe RL, simulation, human oversight

---

### Q20: How do you debug an RL agent that isn't learning?

**Answer:**

**Debugging checklist:**

1. **Check environment:**
   - Can agent reach goal states?
   - Are rewards scaled appropriately?
   - Is state representation informative?

2. **Verify algorithm:**
   - Is update rule implemented correctly?
   - Are hyperparameters reasonable?
   - Is exploration sufficient?

3. **Test on simple tasks:**
   - Can agent learn trivial problems?
   - Gradually increase complexity

4. **Monitor metrics:**
   - Average reward (should increase)
   - Q-values (should stabilize)
   - Exploration rate (should decrease)
   - Episode length (should decrease for goal tasks)

5. **Compare to baseline:**
   - Does agent beat random policy?
   - Try simpler algorithm first

---

## Advanced Topics

### Q21: What is experience replay and why is it useful?

**Answer:**

**Experience replay** stores past experiences (s, a, r, s') in a buffer and samples random batches for training.

**Benefits:**
1. **Break correlations**: Sequential experiences are correlated; random sampling breaks this
2. **Sample efficiency**: Reuse experiences multiple times
3. **Stability**: Smooths out learning updates

**Used in**: DQN and other deep RL algorithms

**Implementation**: Store transitions in circular buffer, sample uniformly or prioritized

---

### Q22: What is the difference between model-free and model-based RL?

**Answer:**

**Model-free RL (Q-Learning, PPO)**:
- Learns policy/value directly from experience
- Doesn't learn environment dynamics
- More sample efficient for simple environments
- Examples: Q-Learning, SARSA, DQN, PPO

**Model-based RL (Dyna-Q, MCTS)**:
- Learns model of environment (transition function)
- Uses model for planning/simulation
- More sample efficient for complex environments
- Can plan ahead without real interaction
- Examples: Dyna-Q, MBPO, AlphaGo (MCTS)

**Tradeoff**: Model-based can be more efficient but requires accurate model

---

### Q23: Explain the Bellman equation.

**Answer:**

The **Bellman equation** expresses the recursive relationship between current and future values.

**For V(s)**:
```
V(s) = E[r + γV(s')]
```
"Value of current state = immediate reward + discounted value of next state"

**For Q(s,a)**:
```
Q(s,a) = E[r + γ max Q(s',a')]
```
"Value of action = immediate reward + discounted max value of next state"

**Bellman optimality equation**: Characterizes optimal value function

**Importance**: Foundation of many RL algorithms (value iteration, Q-Learning)

---

### Q24: What are some advanced RL algorithms?

**Answer:**

**Deep RL:**
- **DQN**: Q-Learning with neural networks
- **Double DQN**: Reduces overestimation in DQN
- **Dueling DQN**: Separate value and advantage streams

**Policy Gradient:**
- **REINFORCE**: Basic policy gradient
- **A3C**: Asynchronous advantage actor-critic
- **PPO**: Proximal policy optimization (state-of-the-art)
- **TRPO**: Trust region policy optimization

**Model-Based:**
- **Dyna-Q**: Combines learning and planning
- **MBPO**: Model-based policy optimization

**Multi-Agent:**
- **MADDPG**: Multi-agent DDPG
- **QMIX**: Value decomposition for cooperation

---

## Problem-Solving Questions

### Q25: You have an RL agent that converges to a suboptimal policy. What could be wrong?

**Answer:**

**Possible causes:**

1. **Insufficient exploration:**
   - ε decayed too quickly
   - Solution: Slower decay, higher ε_min

2. **Poor reward design:**
   - Rewards don't align with objective
   - Solution: Redesign rewards, add shaping

3. **Local optimum:**
   - Agent stuck in local optimum
   - Solution: Increase exploration, restart training

4. **Learning rate issues:**
   - α too high (unstable) or too low (slow)
   - Solution: Tune learning rate

5. **Discount factor:**
   - γ too low (short-sighted)
   - Solution: Increase γ

6. **State representation:**
   - Missing important information
   - Solution: Add features to state

---

### Q26: How would you optimize test flow selection using RL?

**Answer:**

**Problem formulation:**

1. **State**: Binary vector of tests performed
2. **Action**: Select which test to perform next (0-9)
3. **Reward**: 
   - Large positive for detecting defect early
   - Penalty for test costs
   - Large negative for missing defects

4. **Algorithm**: Q-Learning (small discrete state space)

5. **Training**: 
   - Simulate on historical chip data
   - Train for 1000+ episodes
   - Use ε-greedy exploration

6. **Evaluation**:
   - Compare to random policy
   - Measure: detection rate, cost, tests used

**Expected improvement**: 30-50% cost reduction while maintaining/improving detection rate

---

### Q27: Explain how you would implement Q-Learning from scratch.

**Answer:**

**Implementation steps:**

```python
# 1. Initialize Q-table
Q = defaultdict(lambda: np.zeros(num_actions))

# 2. Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # 3. Select action (ε-greedy)
        if random.random() < epsilon:
            action = random.choice(available_actions)
        else:
            action = argmax(Q[state])
        
        # 4. Take action, observe result
        next_state, reward, done = env.step(action)
        
        # 5. Update Q-value
        td_target = reward + gamma * max(Q[next_state])
        td_error = td_target - Q[state][action]
        Q[state][action] += alpha * td_error
        
        state = next_state
    
    # 6. Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

**Key components**: Q-table, ε-greedy, TD update, epsilon decay

---

## Summary

This guide covers fundamental RL concepts, Q-Learning specifics, comparisons with other methods, practical applications, and problem-solving approaches. Understanding these topics prepares you to discuss RL confidently in technical settings.

**Key takeaways:**
- RL learns from interaction and rewards
- Q-Learning learns optimal action values
- Balance exploration and exploitation
- Design rewards carefully
- Debug systematically

**Practice**: Implement Q-Learning, explain concepts in your own words, apply to different problems.

---

**Good luck with your technical discussions!** 🚀
