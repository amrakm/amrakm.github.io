---
layout: single
title: "Monte_Carlo"
mathjax: true
tags:
    - python
    - notebook
--- 
# Monte Carlo Methods

This notebook is an implementation of Monte Carlo (MC) algorithms that is
introduced in Richard Sutton's book applied to Blackjack environment of OpenAI's
gym. I used Udacity's visualisation function for ploting

With MC we don't assume full knowledge of the environment. We only need
experience-sample sequences of (states, actions, reward)  from actual or
simulated interaction with an environment. This requires no prior knowledge of
the environment's dynamics. We can approximate the model by generating samples
of transitions according to a desired probability distribution, but we can't
obtain the distribution in explicit form as required for dynamic programming.


Monte Carlo methods in this context are ways of solving the reinforcement
learning problem based on averaging sample returns.


General Policy Iteration (GPI):

Policy Evaluation $\leftrightarrow$ Policy Improvement


Monte Carlo prediction

We learn the state-value function of a given policy by interacting with the
environment (collecting samples of interactions) using a that policy.

Policy Improvement:



Control



Advantages of MC over DP:
* It doesn't require knowledge of the environment model.
* Unlike DP it doesn't rely on bootstrapping, thus, values of states are
estimated independently and we can learn values of subset of states if we always
start sampling from these states.
* It can learn from actual experience or simulated one?
* Less harmed by violations of the Markov property


> In control methods we are particularly interested in approximating action-
value functions,
because these can be used to improve the policy without requiring a model of the
environment’s transition dynamics

 
 
Monte Carlo estimate of stat-action values can't use determinstic policies as it
will never visit some trajectories, so we add an element of exploration. 
 
 The first-visit MC method estimates $v_{π}(s)$
as the average of the returns following first visits to s 
 
every-visit MC method averages
the returns following all visits to s. 

**In [16]:**

{% highlight python %}
import sys
import gym
import numpy as np
from collections import defaultdict
from collections import Counter
from plot_utils import plot_blackjack_values, plot_policy
{% endhighlight %}
 
#### First-visit MC prediction, for estimating $V ≈ vπ$
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

Source:
[Richard S. Sutton](http://incompleteideas.net/book/bookdraft2017nov5.pdf) 
 
### Creating an instance of the [Blackjack](https://github.com/openai/gym/blob/m
aster/gym/envs/toy_text/blackjack.py) environment. 

**In [5]:**

{% highlight python %}
env = gym.make('Blackjack-v0')
{% endhighlight %}

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
 
### Part 1: MC Prediction: (estimating the action-value function)


Definitions:

`env`: This is an instance of OpenAI Gym's Blackjack environment.
`episode`: This is a list of (state, action, reward) tuples (of tuples) and
corresponds to $(S_0, A_0, R_1, \ldots, S_{T-1}, A_{T-1}, R_{T})$, where $T$ is
the final time step.
`Q`: A dictionary (of one-dimensional arrays) where `Q[s][a]` is the estimated
action value corresponding to state `s` and action `a`.
`gamma`: The discount rate.  It must be a value between 0 and 1, inclusive
(default value: `1`). In this case we will select `1` as it is an episodic
environment and we care more about the reward in the final step.
`alpha`: This is the step-size parameter for the update step.
 
 
Will start by defining a random policy to explore the environment, then we will
evaluate the `Q` (state-action) values based on this policy 

**In [688]:**

{% highlight python %}
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
{% endhighlight %}
 
### Updating Q values Using Incremental Mean

<div> 
    $Q(S_{t}, A_{t}) \leftarrow Q(S_{t}, A_{t}) + \frac{1}{N(S_{t}, A_{t})}(G_{t} -
Q(S_t - A_t))$
</div>

We update Q value after each episode by the error between estimated return
compared to stored Q value for that state-action averaged by the number of times
we visited this state-action pair before:

$\delta_t = (G_{t} - Q(S_t, A_t))$ 

**In [716]:**

{% highlight python %}
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
{% endhighlight %}
 
### Let's evaluate the state-action value ($Q$) using random policy 

**In [None]:**

{% highlight python %}
# obtain the action-value function
Q = mc_prediction_q(env, 50000, random_policy)
{% endhighlight %}
 
#### We can visualise Q function in a 3D plot

Plotting the state-action value in two plots as each state consist of three
variables : Player's Current Sum - Dealer's Showing Card - Usable Ace 

**In [717]:**

{% highlight python %}
# obtain the corresponding state-value function
V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
         for k, v in Q.items())

plot_blackjack_values(V_to_plot)
{% endhighlight %}

    Episode 50000/50000.

 
![png]({{ BASE_PATH }}/images/monte_carlo_17_1.png) 

 
### Part 2: MC Control


Your algorithm has four arguments:
- `env`: This is an instance of an OpenAI Gym environment.
- `num_episodes`: This is the number of episodes that are generated through
agent-environment interaction.
- `alpha`: This is the step-size parameter for the update step.
- `gamma`: This is the discount rate.  It must be a value between 0 and 1,
inclusive (default value: `1`).

The algorithm returns as output:
- `Q`: This is a dictionary (of one-dimensional arrays) where `Q[s][a]` is the
estimated action value corresponding to state `s` and action `a`.
- `policy`: This is a dictionary where `policy[s]` returns the action that the
agent chooses after observing state `s`.

 
 
$π(s).= arg max_{a}q(s, a)$ 

**In [6]:**

{% highlight python %}
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
{% endhighlight %}

**In [7]:**

{% highlight python %}
def evaluate_Q(env, Q, n_episodes=1000):
    
    status_map = {-1:'loss', 0:'draw', 1:'win'}

    average_reward = []
    for _ in range(n_episodes):

        average_reward.append(generate_episode_from_Q(env, Q, epsilon=0, n_actions=env.action_space.n)[-1][-1])

    return {status_map[k]:v for k,v in Counter(average_reward).items()}
{% endhighlight %}
 
#### MC-Control constant-alpha


We would like to give more weights to returns estimated recently more than ones
estimated at the first few episodes as we expect the recent ones to be generated
from a better policy. Thus, we replace the weighting average with a constant
alpha that determines how much we want to emphasize later returns and how much
we forget from the past

$Q(S_{t}, A_{t}) \leftarrow Q(S_{t}, A_{t}) + \alpha (G_{t} - Q(S_t, A_t))$

We can re-write this equation as:

$Q(S_t,A_t) \leftarrow (1-\alpha)Q(S_t,A_t) + \alpha G_t$

- If $\alpha=0$, then the action-value function estimate is never updated by the
agent.
- If $\alpha = 1$, then the final value estimate for each state-action pair is
always equal to the last return that was experienced by the agent (after
visiting the pair). 

**In [11]:**

{% highlight python %}
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
{% endhighlight %}

**In [9]:**

{% highlight python %}
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


{% endhighlight %}

**In [32]:**

{% highlight python %}
# obtain the estimated optimal policy and action-value function
policy, Q = mc_control(env, num_episodes=1000000, alpha=0.02, gamma=1.0,
                       epsilon_start=1., epsilon_decay=0.99999, epsilon_min=0.05)
{% endhighlight %}

    Episode 1000000/1000000.

**In [33]:**

{% highlight python %}
evaluate_Q(env, Q)
{% endhighlight %}




    {'draw': 81, 'win': 413, 'loss': 506}


 
### Plotting the corresponding state-value function. 

**In [34]:**

{% highlight python %}
# obtain the corresponding state-value function
V = dict((k,np.max(v)) for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V)
{% endhighlight %}

 
![png]({{ BASE_PATH }}/images/monte_carlo_28_0.png) 

 
Finally, we visualize the policy that is estimated to be optimal. 
 
Epsilon greedy.

$
\pi(a|s) \longleftarrow
\begin{cases}
\displaystyle 1-\epsilon +\epsilon/|\mathcal{A}(s)|&amp;amp; \textrm{if
}a\textrm{ maximizes }Q(s,a)\\
\displaystyle \epsilon/|\mathcal{A}(s)| &amp;amp; \textrm{else}
$ 
 


$\pi(a|s) \longleftarrow
\displaystyle 1-\epsilon +\epsilon/|\mathcal{A}(s)| \textrm{if }a\textrm{
maximizes }Q(s,a) \\
\epsilon \mathcal{A}(s)|  \textrm{else} $
 

**In [35]:**

{% highlight python %}
# plot the policy
plot_policy(policy)
{% endhighlight %}

 
![png]({{ BASE_PATH }}/images/monte_carlo_32_0.png) 

 
The **true** optimal policy $\pi_*$ can be found in Figure 5.2 of the
[textbook](http://go.udacity.com/rl-textbook) (and appears below).
![True Optimal Policy](images/optimal.png) 

**In [206]:**

{% highlight python %}
!open file:///Users/amr/Downloads/nd893/Deep%20Reinforcement%20Learning%20Nanodegree%20v2.0.0/Part%2001-Module%2001-Lesson%2008_Monte%20Carlo%20Methods/img/screen-shot-2018-05-04-at-2.49.48-pm.png

{% endhighlight %}
 
![algo](file:///Users/amr/Downloads/nd893/Deep%20Reinforcement%20Learning%20Nano
degree%20v2.0.0/Part%2001-Module%2001-Lesson%2008_Monte%20Carlo%20Methods/img/sc
reen-shot-2018-05-04-at-2.49.48-pm.png "ShowMyImage") 
 
 Off - Policy Learning

> The on-policy approach in the preceding section is actually a compromise—it
learns action values not for the optimal policy, but for a near-optimal policy
that still explores. A more
straightforward approach is to use two policies, one that is learned about and
that becomes the optimal
policy, and one that is more exploratory and is used to generate behavior. The
policy being learned
about is called the target policy, and the policy used to generate behavior is
called the behavior policy.
In this case we say that learning is from data “off” the target policy, and the
overall process is termed
off-policy learning.

>  Off-policy methods require additional concepts
and notation, and because the data is due to a different policy, off-policy
methods are often of greater
variance and are slower to converge. On the other hand, off-policy methods are
more powerful and
general.

> Off-policy methods also have a variety of additional uses in applications. For
example,
they can often be applied to learn from data generated by a conventional non-
learning controller, or
from a human expert. 
