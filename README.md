<div align=center>
  <h1>
    Markov Decision Process & Dynamic Programming
  </h1>
  <p>
    <b>KAIST CS470: Introduction to Artificial Intelligence (Spring 2023)</b><br>
    Programming Assignment 3
  </p>
</div>

<div align=center>
  <p>
    Instructor: <a href=https://sites.google.com/site/daehyungpark target="_blank"><b>Daehyung Park</b></a> (daehyung [at] kaist.ac.kr)<br>
  </p>
</div>


## Assignment 3
### Problem 1: Markov Decision Process [50 pts]
In this section, you design a Markov decision process (MDP) for a toy environment, called <i>gridworld</i>, which is often used for reinforcement learning (RL). The environment is a stochastic version of the pre-built discrete <i>gridworld</i> environment from <i>OpenAI Gym</i>. In order to represent a task in the environment, you define an MDP, $< S,A,T,r,γ >$, where $S$, $A$, $T$, $r$ ,and $γ$ are states, actions, transition probabilities, a reward function, and a discount factor, respectively. We particularly use a 8 × 10 size of <i>gridworld</i> environment, where the coordinate of left-top and right-bottom cells are [0,0] and [7,9], respectively. An agent can move onto one of the four nearest cells or stay. Please, fill your code in the blank section following the <b>“PLACE YOUR CODE HERE”</b> comments in the <b>CS470_Assignment3_problem.ipynb</b> file following the subproblems below.


<b>TRANSITION MODEL</b>: Implement a stochastic <b>transition_model()</b> of the environment dynamics. You need to fill out the <b>transition_model()</b> function, which returns a list of transition probabilities over the next states given a state and an action. In order to define the transition model, you have to consider following rules:

- The agent has five possible movements: stay, up, down, left, and right (see Fig.1),
- The agent is not allowed to move off the grid or move on an obstacle (grey); If the agent tries to move off the grid or move on an obstacle, it will end up staying at the previous cell,
- The agent applies a selected action with a probability $1 − ε$,
- The agent (uniform) randomly applies an action among unselected actions with a (noise) probability $ε/4$,
- The agent terminates the episode when it reaches goal or trap (red) cells, and
- The agent cannot leave goal or obstacle cells.

Note that we use predefined $ε = 0.05$ on the assignment IPython notebook.

<img src="/Figure/Figure1.png" width="50%" height="50%">

Figure 1: An exemplar <i>gridworld</i> environment with available actions. An agent cannot move on an obstacle (grey). An episode ends when the agent reaches either a trap (red) cell or a goal (green) cell. In this environment, the cell with index 1 and 78 corresponds to [1, 0] and [6, 9], respectively.



<b>REWARD FUNCTION</b>: Design a reward function by filling out the <b>compute_reward(s,a,s’)</b> function, where the arguments represent a current state, an action, and a next state. The function returns a positive reward <i>+5</i> when the agent reaches a goal (terminal state in the code) , a negative reward <i>-10</i> when the agent reaches a trap, otherwise <i>0</i>. In addition, the function provides a step penalty of <i>−0.1</i> given any action.


<b>STEP FUNCTION</b>: Implement the <b>step</b> function that takes an action and applies it to the environment. The applied action leads to a stochastic transition to the next state. In details, the <b>step</b> function performs followings:
- sample a next state according to the transition function,
- calculate a reward value,
- update the current state to the next state, and
- return a result tuple, (next state, reward, termination signal, information).


<b>TERMINATION</b>: We terminate an episode when an agent is in either goal or trap cells. To deal with the episode termination conditions, you have to implement the <b>is_done()</b> function that returns a Boolean termination signal that is whether the current episode has to be terminated or not. Finally, you are ready to make your agent interact with the environment for RL. 


On your report,
```
1. [15 pts] print out the transition probabilities to all the next states given
  • a current state [3, 1] and a selected action “Down”,
  • a current state [0, 6] and a selected action “Right”,
  • a current state [3, 5] and a selected action “Right”,
2. [5 pts] plot the resulting histogram of returns produced by a dummy policy in the IPython notebook for 100 episodes,
3. [5 pts] plot the distribution of trajectories produced by a dummy policy in the IPython notebook for 300 episodes,
4. [25 pts] attach your implemented code from <b>transition_model()</b>, <b>compute_reward()</b>, <b>step()</b>, and <b>is_done()</b> functions.
```

### Problem 2: Dynamic Programming [50 pts]
In this problem, you implement and analyze a representative DP algorithm: value iteration (VI).

#### 2.1. Value Iteration (VI) [30 pts]
Implement the VI algorithm for the stochastic gridworld environment by filling out the ValueIteration class. The VI algorithm iteratively and directly updates each state value $V_t(s)$ at a time step t given a transition model $T$:

<img src="/Figure/equation1.png" width="50%" height="50%">

VI stops to update the values when the maximum update error $∆$ is lower than a certain threshold $θ$, where $∆ ← max(∆, ∥Vt+1(s) − Vt(s)∥) ∀s ∈ S$. (See details on the RL book p.83.) 

On your report, please
```
• write down the state values of the first 8 states of the gridworld environment1,
• overlay the best action at each state based on the state-action values,
• plot the distribution of trajectories produced by the trained policy for 100 episodes, and
• attach your implemented code from value iteration() and get action() functions on your report.
```

#### 2.2. Comparison under different transition models [20 pts]
Suppose that the probability of taking a random action is 10% (i.e., $ε = 0.1$). This transition model will affect the <b>exploration</b> process of <b>VI</b>. Thus, in this part, you are asked to compare/analyze the effects when you run <b>VI</b> with $ε = 0.1$ and $ε = 0.4$. 

On your report, please
```
• plot the expected returns per ε value with respect to the number of iterations until convergence (two graphs or one unified graph),
• overlaying the best actions at each state per ε value (two visualizations),
• plot the distribution of trajectories produced by the trained policy for 100 episodes per ε value (two visualizations), and
• compare/analyze the effect of different ε based on the above results.
```
<i>Note that analysis does not mean you have to write down a long report. Scientific writing requires a concise delivery of your thoughts/results.</i>


### Results
#### 1.1 Convolution and Average Pooling using NumPy
As same reason, results of Filter 2 (takes other values on vertical line) and Filter 3(takes other values on other diagonal) look spread in vertical and reverse diagonal direction respectively.
In case of Average Pooling, it makes 4 parameters to 1 parameter by averaging them. That means the image information is compressed and also lost. That is reason why the results look more blurry.

<center><img src="/Figure/visualization1.png" width="50%" height="50%"></center>

<center><img src="/Figure/visualization2.png" width="50%" height="50%"></center>

  
#### 1.2 Convolution and Average Pooling using PyTorch
I define the error between two kinds of methods as euclidean distance between two results. The euclidean distance between numpy output and pytorch output by each example and each filter can be shown below. we can know that both are almost same.

<center><img src="/Figure/visualization3.png" width="50%" height="50%"></center>

<center><img src="/Figure/distance.png" width="50%" height="50%"></center>

#### 2.1 A CNN with MaxPooling layers
Let the input image size as (1, 28, 28).
- (Layer 1) Conv2d-1 (32, 3, 3) → 32 * 3 * 3 + 32 = 320 parameters
- (Layer 3) Conv2d-2 (64, 3, 3) → 64 * 32 * 3 * 3 + 64 = 18,496 parameters
- (Layer 6) Linear-1 (64, 30976) → 64 * 30976 + 64 = 1,982,528 parameters
- (Layer 7) Linear-2 (32, 64) → 32 * 64 + 32 = 2,080 parameters
- (Layer 8) Linear-3 (10, 32) → 10 * 32 + 10 = 330 parameters
Number of total parameters is 2,003,754.

<center><img src="/Figure/training_history.png" width="50%" height="50%"></center>

#### 2.2 Prevention of Overfitting
I used two methods(weight decay and dropout) to prevent overfitting. As we can see from training-validation accuracies and loss graph, the gap of accuracies and loss between training and validation with two methods is less than the gap without them. Also, the accuracy on the test images is quite increased.

<center><img src="/Figure/training_history2.png" width="50%" height="50%"></center>

The below explains why each two methods can prevent overfitting.
- Weight decay(1e-5)
  - By using Weight decay to set the lower bound of loss value, parameters can be prevented from being excessively      updated (i.e., overfitting).

- Drop out(p = 0.7)
  - Dropout prevents the corresponding parameter from being updated at that point by making certain parameters zero with the probability of p. That is, the number of times parameters are updated is reduced in probability, preventing overfitting.

#### 3. Comparison of MLP and CNN
For reasonable comparison, I tried to make numbers of total trainable parameters of both model same. The CNN model has 2,003,754 parameters and the MLP model has 2,005,610 parameters. Although the number of parameters are similar, the accuracy on test images of the CNN and MLP are 91.48% and 80.54%, respectively. I think the reason why the accuracy gap exists is affect of locality. The CNN model can consider locality information of input images. In the other hands, the MLP model can’t refer their locality information because they make input images flatten immediately when they come in the model. That is reason why CNN is powerful than MLP in image task.

<center><img src="/Figure/comparison.png" width="50%" height="50%"></center>

# ETC
For educational purpose only. This software cannot be used for any re-distribution with or without modification. The lecture notebook files are copied or modified from the material of Siamak Ravanbakhsh. 

