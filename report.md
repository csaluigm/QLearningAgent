# Code Structure

At the root level we find the markdown files readme and report that will help you to understand how the project works.
The UnityEnv folder contains the windows version of the unity banana env. It wouldn't be necessary in the repo it can be downloaded separately but this way is quicker. Like some "portable" version. Add here the specific version for your S.O if you aren't using Windows.

we can also find some assets for the readme.

The source folder contains all the python files to code the agent and the rest of the required logic.

    - model contains the Torch implementation of the neural network model that is being used in the project.
    - cli.py has a bit of code to provide the command line interface argument "--train" this way we can set dynamic logic based on terminal's input
    - config.py Is basically a wrapper class for the hyperparameters so it doesn't make other files bigger and keeps our code clean.
    - dqn_agent_bananas.py Contains the core of the project, where we use the previously defined torch model to learn from the experiences. The code is quite similar to the dqn exercices of the course but logically adapted for this problem.
    - replay_buffer.py It contains the memory experience replay isolated just to be cleaner.
    
# Learning Algorithm

# Plot of Rewards

![](reward_plot.jpeg)

# Ideas for Future Work

To improve the agent's performance we could do hyperparameter optimization using some framework like Optuna to make the model converge faster or even collect better experiences for its learning process. We can think about searching for a better value of each of the config.py values or also tune the neural network itself. More layers, More units.

We can also use DQN improvements as DDQN and/or prioritized experienced replay.