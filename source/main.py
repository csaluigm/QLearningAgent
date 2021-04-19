from  dqn_agent_bananas import Agent
from collections import deque
from cli import Cli
import numpy as np
import torch
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

cli = Cli()
args = cli.parse()

env = UnityEnvironment(file_name="./UnityEnv/Banana.exe")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]


def train(episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    agent = Agent(state_size=37, action_size=4, seed=0)
    scores = []                      
    scores_window = deque(maxlen=100)  
    eps = eps_start                   

    for i_episode in range(1, episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for _ in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(int(action))[brain_name]       
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 

        scores_window.append(score)      
        scores.append(score)              
        eps = max(eps_end, eps_decay*eps) 

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.1:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'solved_score_13.1.pth')

        torch.save(agent.qnetwork_local.state_dict(), 'trained_model.pth')
    return scores

def print_env_info(env):
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

def plot_score_chart(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

if args.train:
    scores = train()
else:
    scores = test()

print_env_info(env)
scores = train()
plot_score_chart(scores)

    
