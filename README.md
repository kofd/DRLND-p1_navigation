[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

## Project Details

The environment requires the following libraries:

```
unityagents
gym
torch
numpy
matplotlib
```

Using Navigation.ipynb, the environment can be learned. It iterates though the environment for 1000
episodes with 1000 timesteps per episode, and is somewhat solved after the agent averages over a score
of 14 or reaches the end of training with a average score over 10. It uses the observation and 
vector action space, but the Navigation_Pixels.ipynb is available for training on the images.

## Getting Started

The required libraries can be installed with:

```bash
pip install unityagents gym torch numpy matplotlib
```

and the environment can be downloaded from the instructions in the following location:
https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation

## Running the code

The code can be run by launching a jupyter notebook and opening Report.ipynb. Avoid running the code
in section 3 to avoid closing the environment. If you would like to avoid running the training, simply run
the final block after setting up the environment variables and unity environment.

## Improvements

The Pixels code hasn't been run yet but the code exists, with a 2 line drop-in to switch to a
convolutional model. Double DQN, prioritized replay, and Dualing DQN have all been implemented
but there are still other improvements that could be added from rainbow such as Noisy DQN, multi-step
bootstrap targets, and distributional DQN.
