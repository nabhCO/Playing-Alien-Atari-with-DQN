# README
This is my final project for CSPB 3202 (Introduction to Artificial Intelligence). The goal of this project was to try and train a DQN agent to play the Alien Atari game, a Gymnasium environment.

# Repository File Guide

- alien_dqn.py: Run the environment and agent from this file.

- class_defs: Contains class definitions for the CNN and Replay Buffer, as well as the definition for the Experience named tuple

- hyperparameters: Change hyperparameter values here, as well as number of samples in a batch and number of episodes for training

- util: Contains functions for transforming observations to a Tensor object with the correct dimensions, and a function for graphing average rewards over the training period
