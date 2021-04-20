# QLearningAgent
DQN Agent for Unity Banana navigation environment

![](agent.gif)
# Project Details
The environment of this project has four possible actions
    -W move forward (or do nothing as it is the default behavior).
    -S move backward.
    -A turn left.
    -D turn right.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.

The environment reward system is quite simple, +1 for retrieving yellow bananas -1 punishment for picking blue ones.o

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

# Getting Started

Make sure you have python installed in your computer. (The project was created with python 3.6.12) [download python](https://www.python.org/downloads/)

Navigate to the root of the project:

`cd QLearningAgent` 

Install required python packages using pip or conda, for a quick basic setup use:

`pip install -r requirements.txt` 

# Instructions

You can run the project from some Editor like VS code or directly from commandline:

`python main.py --train 1`

This will train the agent and will store 2 model weights. One when it pass the environment solved condition and other after 1800 episodes.
The proposed code normally solves the environment around 600 episodes.

Once the model is trained you can check its behavior by testing it:

`python main.py`