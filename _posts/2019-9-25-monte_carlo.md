---
layout: single
title: "Monte Carlo in the Context of Reinforcement Learning"
mathjax: true
tags:
    - python
    - notebook
    - reinforcement learning
--- 

In this notebook I will share my study notes along with implementation of few Monte Carlo (MC) algorithms that is introduced in [Richard Sutton's book](http://incompleteideas.net/book/bookdraft2017nov5.pdf) applied to Blackjack environment of [OpenAI's gym](https://github.com/openai/gym).

------------



## Introduction:

The aim of reinforcement learning is to find the optimal policy that maximises the agent’s reward in a given environment.

The optimal policy should maximise the accumulated reward starting from any state in environment. So we want to find the policy the maximises the state value function or the state-action value fucntion as the following:



$$ v_{*}(s) = \max\limits_{a} \sum\limits_{s^\prime, r} p(s^{\prime},r | s, a)[r + \gamma v_{*}(s^\prime)] $$

$$ q_{*} (s, a) = \sum\limits_{s^{\prime}, r} p(s^{\prime}, r | s, a)[r + \gamma  \max\limits_{a^{\prime}} q_{*}(s^{\prime}, a^{\prime})]  $$

In order to calculate this analytically, we will need to sum over all possible states, which is not possible when we have a large number of states. We can get around this by using an iterative approach like Dynamic Programming(DP) or Monte Carlo. 

Main differences between Dynamic Programming(DP) and Monte Carlo (MC):

* Dynamic programming requires full knowledge of the transition model of the environment and the reward system to evaluate the policy.
* Monte Carlo don't assume full knowledge of the environment. We only need experience-sample sequences of (states, actions, reward)  from actual or simulated interaction with an environment. This requires no prior knowledge of the environment's dynamics. We can approximate the model by generating samples of transitions according to a desired probability distribution, but we can't obtain the distribution in explicit form as required for dynamic programming.


Advantages of MC over DP:
* It doesn't require knowledge of the environment model.
* Unlike DP it doesn't rely on bootstrapping, thus, values of states are estimated independently and we can learn values of subset of states if we always start sampling from these states.
* It can learn from actual experience or simulated one.
* Less harmed by violations of the Markov property

One thing to keep in mind for implementation:
* Monte Carlo estimate of stat-action values can't use determinstic policies as it will never visit some trajectories, that's why we add an element of stochasticity in the exploration policy.

 
In short:
> Monte Carlo methods in this context are ways of solving the reinforcement learning problem based on interacting with the environment and averaging sample returns.          
       
       




--------


### Implementation: 
The main algorithm (MC control) is devided into two building blocks. 

#### Monte Carlo Control using General Policy Iteration (GPI):
We iterate between: Policy Evaluation and Policy Improvement
Until the policy converges to the optimal policy or until it stabilizes and stops improving.


- #### Monte Carlo prediction:
We learn the state-value function of a given policy by interacting with the environment (collecting samples of interactions) using a that policy.

- #### Monte Carlo Policy Improvement:
We update the policy to use best action according the updated estimate of state-action values.

![policy iteration](/images/policy_iteration.png)
-----------------------



```python
import sys
import gym
import numpy as np
from collections import defaultdict
from collections import Counter
from plot_utils import plot_blackjack_values, plot_policy
```

## [Blackjack](https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py) environment:


```python
env = gym.make('Blackjack-v0')
```

    /anaconda3/envs/drlnd/lib/python3.6/site-packages/gym/envs/registration.py:14: PkgResourcesDeprecationWarning: Parameters to load are deprecated.  Call .resolve and .require separately.
      result = entry_point.load(False)


Each state is a 3-tuple of:
- the player's current sum $\in \{0, 1, \ldots, 31\}$,
- the dealer's face up card $\in \{1, \ldots, 10\}$, and
- whether or not the player has a usable ace (`no` $=0$, `yes` $=1$).

The agent has two potential actions:

```
    STICK = 0
    HIT = 1
```

## MC Prediction: (estimating the action-value function)

Definitions:

`env`: This is an instance of OpenAI Gym's Blackjack environment.     
`episode`: This is a list of (state, action, reward) tuples (of tuples) and corresponds to $(S_0, A_0, R_1, \ldots, S_{T-1}, A_{T-1}, R_{T})$, where $T$ is the final time step.      
`Q`: A dictionary (of one-dimensional arrays) where `Q[s][a]` is the estimated action value corresponding to state `s` and action `a`.    
`gamma`: The discount rate.  It must be a value between 0 and 1, inclusive (default value: `1`). In this case we will select `1` as it is an episodic environment and we care more about the reward in the final step.     
`alpha`: This is the step-size parameter for the update step.


#### First-visit MC prediction, for estimating $V ≈ vπ$


The first-visit MC method estimates $v_{π}(s)$
as the average of the returns following first visits to s

##### Algorithm:

``` 
Initialize:
    π ← policy to be evaluated
    V ← an arbitrary state-value function
    Returns(s) ← an empty list, for all s ∈ S

Repeat forever:
    Generate an episode using π
    For each state s appearing in the episode:
        G ← the return that follows the first occurrence of s
        Append G to Returns(s)
        V (s) ← average(Returns(s))
```   

Source: 
[Richard S. Sutton](http://incompleteideas.net/book/bookdraft2017nov5.pdf)

#### Updating Q values Using Incremental Mean 

$$Q(S_{t}, A_{t}) \leftarrow Q(S_{t}, A_{t}) + \frac{1}{N(S_{t}, A_{t})}(G_{t} - Q(S_t - A_t))$$
    
We update Q value after each episode by the amount of error between estimated return compared to stored Q value for that state-action, averaged by the number of times we visited this state-action pair before: 

$$\delta_t = (G_{t} - Q(S_t, A_t))$$

Will start by defining a random policy to explore the environment, then we will evaluate the `Q` (state-action) values based on this policy


```python
def random_policy(env, state=None):
    return np.random.choice(env.action_space.n, 1)[0]

def generate_episode_from_policy(env, policy):
    '''
    env: 
    policy:
    '''
    
    episode = [] 
    state = env.reset()
    
    while True:
        action = policy(env, state)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
            
    return episode
```


```python
def mc_prediction_q(env, num_episodes, policy, gamma=1.0):
    
    # initialize empty dictionaries of arrays
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            
        episode = generate_episode_from_policy(env, policy)
        states, actions, rewards = zip(*episode)
        
        discount_factors = [gamma**x for x in range(len(rewards)+1)]

        first_visit_flag = {(x,y):1 for x,y in zip(states, actions)}
        
        for idx, (state, action) in enumerate(zip(states, actions)):
            
            if first_visit_flag[(state,action)] == 1:
                
                discounted_rewards = [x*y for x,y in zip(rewards[idx:], discount_factors[:-(1+idx)])]
                sum_discounted_reward = sum(discounted_rewards)
                
                returns_sum[state][action] += sum_discounted_reward
                N[state][action] += 1
                Q[state][action] = returns_sum[state][action] / N[state][action]
            
                first_visit_flag[(state,action)] = 0
    
        
    return Q
```

### Evaluating the state-action value ($Q$) using random policy 


```python
# obtain the action-value function
Q = mc_prediction_q(env, 50000, random_policy)
```

#### Visualising Q function in a 3D plot

Using [Udacity](https://github.com/udacity/deep-reinforcement-learning/blob/master/monte-carlo/plot_utils.py)'s visualisation function for ploting state-action values of Blackjack environment.    
Plotting the state-action value in two plots as each state consist of three variables : Player's Current Sum - Dealer's Showing Card - Usable Ace


```python
# obtain the corresponding state-value function
V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
         for k, v in Q.items())

plot_blackjack_values(V_to_plot)
```



![png](/images/Monte_Carlo_14_1.png)


## MC Control:

The algorithm has four arguments:
- `num_episodes`: This is the number of episodes that are generated through agent-environment interaction.
- `alpha`: This is the step-size parameter for the update step.
- `Q`: This is a dictionary (of one-dimensional arrays) where `Q[s][a]` is the estimated action value corresponding to state `s` and action `a`.
- `policy`: This is a dictionary where `policy[s]` returns the action that the agent chooses after observing state `s`.


#### Algorithm
![MC_GLIE](/images/mc_control_constant_alpha_GLIE.png)


```python
def generate_episode_from_Q(env, Q, epsilon, n_actions):
    
    
    episode = [] 
    state = env.reset()
    
    
    while True:
        
        if state in Q:
        
            ### initialise probabilites of all actions to be epsilon / n_actions
            probability_of_random_action = epsilon /n_actions
            probability_vector = np.ones(n_actions) * probability_of_random_action
            best_action_idx = np.argmax(Q[state])
            probability_vector[best_action_idx] = 1 - epsilon + probability_of_random_action
            
            action = np.random.choice(np.arange(n_actions), p=probability_vector)
            
        else:
            action = env.action_space.sample()
            
            
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        
        if done:
            break
            
    return episode       
```


#### Constant-alpha


We would like to give more weights to returns estimated recently more than ones estimated at the first few episodes as we expect the recent ones to be generated from a better policy. Thus, we replace the weighting average with a constant alpha that determines how much we want to emphasize later returns and how much we forget from the past

$$Q(S_{t}, A_{t}) \leftarrow Q(S_{t}, A_{t}) + \alpha (G_{t} - Q(S_t, A_t))$$



We can re-write this equation as:

$$Q(S_t,A_t) \leftarrow (1-\alpha)Q(S_t,A_t) + \alpha G_t$$    


- If $$\alpha=0$$, then the action-value function estimate is never updated by the agent.
- If $$\alpha = 1$$, then the final value estimate for each state-action pair is always equal to the last return that was experienced by the agent (after visiting the pair).


```python
def update_Q(Q, episode, gamma, alpha, n_actions):


    
    states, actions, rewards = zip(*episode)

    returns_sum = defaultdict(lambda : np.zeros(n_actions))


    discount_factors = [gamma**x for x in range(len(rewards)+1)]

    first_visit_flag = {(x,y):1 for x,y in zip(states, actions)}

    for idx, (state, action) in enumerate(zip(states, actions)):

        if first_visit_flag[(state,action)] == 1:
            
            discounted_rewards = [x*y for x,y in zip(rewards[idx:], discount_factors[:-(1+idx)])]
            sum_discounted_reward = sum(discounted_rewards)
            
            Q_old = Q[state][action]
            delta = sum_discounted_reward -  Q_old
            Q[state][action] = Q_old + alpha * delta

            first_visit_flag[(state,action)] = 0    

    return Q
```


```python
def mc_control(env, num_episodes, alpha, gamma=1.0, epsilon_start=0.99,
               epsilon_decay=0.9, epsilon_min=0.1):
    
    epsilon = epsilon_start
    n_actions = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.ones(n_actions))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        ## updating epsilon and policy before collecting new episode
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
      
        episode = generate_episode_from_Q(env, Q, epsilon, n_actions)
        
        Q = update_Q(Q, episode, gamma, alpha, n_actions)
    
    policy = {k:np.argmax(v) for k,v in Q.items()}
    
    return policy, Q


```


```python
# obtain the estimated optimal policy and action-value function
policy, Q = mc_control(env, num_episodes=1000000, alpha=0.02, gamma=1.0,
                       epsilon_start=1., epsilon_decay=0.99999, epsilon_min=0.05)
```

### Plotting the corresponding state-value function.


```python
# obtain the corresponding state-value function
V = dict((k,np.max(v)) for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V)
```


![png](/images/Monte_Carlo_26_0.png)


Finally, we visualize the policy that is estimated to be optimal.


```python
# plot the policy
plot_policy(policy)
```


![png](/images/Monte_Carlo_28_0.png)


The **true** optimal policy $\pi_*$ can be found in Figure 5.2 of the [textbook](http://incompleteideas.net/book/bookdraft2017nov5.pdf) (and appears below). 
![True Optimal Policy](/images/optimal.png)
