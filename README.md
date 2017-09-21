# Machine Learning Engineer Nanodegree
## Capstone Proposal
Jared Wadsworth  
September 15st, 2017

## Proposal

### Domain Background
Almost since the dawn of Machine Learning and Artificial Intelligence researchers and scientists have been trying to teach machines through Reinforcement Learning (RL). This field of AI was originally inspired by Behaviorist Psychology. Specifically it deals with how an agent chooses actions in an environment to maximize some reward. Generally this problem is captured as a Markov Decision Process which uses a (state, action reward, next_action) tuple. Historically researchers have often used RL to play games such as Chess, Go, or even various Atari games, however some more concrete examples of some applications of RL include: Self Driving Cars, and investing in the Stock Market.

With the advancements made in Deep Learning there has been a renewed interest in RL in what has been called Deep RL. Using these techniques computers have been able to beat the Grandmasters of Chess and Go, and the best gamers at the Atari. It has also made self driving cars a reality.

On a personal note, ever since I first learned about reinforcement learning it has always peaked my interest. To me it is the most fun side of AI. I also believe that it has the greatest potential to change our lives, and help us solve many of the problems of today.

### Problem Statement
_(approx. 1 paragraph)_

https://gym.openai.com/envs/LunarLander-v2/

The problem which I will solve is a game called Lunar Lander. The basic idea of this game is that the lunar lander is coming in for landing, and the agent gets to control its thrusters taking one of four possible actions (left, right, main, none). The agent must land the lander in the specified location on the ground without crashing the lander. Rewards for the agent include landing softly in the specified location (100 - 140 points), crashing (-100 points), landing softly (100 points), Leg ground contact (10 points each), Firing Main engine (-0.3 points per frame). The agent wins if it scores over 200 points.

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

At each step, or frame, of the game the agent receives an array with 8 values corresponding to its location, orientation, and velocity. Combined these are referred to as the "State". Each of these is necessary to allow the agent to navigate down to the landing zone. These inputs come from the OpenAI Gym environment. 

At each frame the agent will take an action, and then receive a reward from the environment (as described above). Using this (state, action, reward, next_state) tuple, the agent is able to determine how effective each action was, and then optimize itself to always choose the best action each frame so that it is able to land softly in the landing zone.

### Solution Statement
_(approx. 1 paragraph)_

An agent is said to have solved this problem if it is able to score an average of greater than 200 points over a period of 100 iterations (games).

### Benchmark Model
_(approximately 1-2 paragraphs)_

A common solution to this type of problem is Q Learning. This however doesn't work because of the continuous values used in the state array it becomes impossible to build a Q table. One solution to this problem is to discretize the state array. By discretizing the state array the agent is able to build a proper Q table and learn a valid policy. This then can be compared to my solution through the metrics described below.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

The metric of success will be the fewest number of iterations (games) until the algorithm solves the problem. For example if it takes algorithm A 10,000 iterations to solve it, and it takes algorithm B 9,000 iterations, then algorithm B is said to be better than algorithm A.

### Project Design
_(approx. 1 page)_

My plan is to start by coding up a simple Q Learner (described in the benchmark session) to use as my benchmark for further testing. After building the benchmark, I will start exploring various other options to see if I can beat it.

I'll then dive into some alternate algorithms such as Deep Q Networks (DQN), Actor-Critic (AC), Asynchronous Actor Critic (A3C), and others. After creating these agents, I'll compare them to the benchmark algorithm as described in the metrics section. After testing each of them, I will select the best one and move on to the next step.

After choosing an algorithm, I'll continue by altering the things the algorithm learns on. I'll potentially introduce some artificial rewards, or alter the rewards before the agent learns on them. Then I might change some of the items in the state to see if I can get it to learn better, each time comparing it to the previously selected agent.

Finally I'll run the new agent, and compare it to the benchmark to see how much better it performed. I will then submit my findings, and code to both Udacity and OpenAI.
