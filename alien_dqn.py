import gymnasium as gym
import torch
import numpy as np
from util import transform_obs
import hyperparameters
from class_definitions import ConvNeuralNet, ReplayBuffer, Experience
from gymnasium.wrappers import ResizeObservation, FrameStackObservation
from ale_py import ALEInterface


'''
SETUP:
These are the steps for setting up the environment, along with everything we need to train our agent:
-The CNN (model)
-Optimizer (optimizer, using RMSprop and parameters.alpha as learning rate)
-Replay Buffer (replay_buffer, for storing and retrieving agent experiences in the environment)
'''
ale = ALEInterface()

#setting up environment and preprocessing- observation are image stacks of 4 most recent observations that are grayscale, 84x84
env = gym.make('ALE/Alien-v5', obs_type="grayscale", render_mode="human") #change render_mode while training
env = ResizeObservation(env, (84, 84))
env = FrameStackObservation(env, 4) # return stack of most recent 4 observations


model = ConvNeuralNet()
optimizer = torch.optim.RMSprop(model.parameters(), lr=hyperparameters.alpha)
replay_buffer = ReplayBuffer(500)

'''
HELPER FUNCTIONS:
Helper functions for exploration, exploitation and training.
'''

'''
choose_action(obs, epsilon)
Parameters:
-obs (current observation from environment)
-epsilon (probability that we will choose a random action (exploration) over one chosen by our model (exploitation))

Description:
Sets up the probabilities of exploration and exploitation options, randomly chooses one.
Calculates new value for epsilon after decay
If we choose exploration, randomly sample an action from our environment's action space.
If we choose exploitation, pass observation stack through model, select index of maximum value (index of best action). 
Return action and new epsilon
'''
def choose_action(obs, epsilon):

    choose_random_action = np.random.choice(a=[True, False], p=[epsilon, 1-epsilon])

    #exploration
    if choose_random_action:

        action = env.action_space.sample()

        #we're putting action in this form so it matches the model output (with an extra dimension for batch size)
        return torch.tensor([[action]], dtype=torch.long)

    #otherwise exploitation (don't calculate gradients, we're not training the model right now)
    with torch.no_grad():

        action_vals = model.forward(obs)
        return action_vals.max(1).indices.view(1, 1)


'''
Note: This function borrows code pretty heavily from PyTorch's DQN tutorial, their data handling for batches and processing
is super efficient (and I'm very new to PyTorch). Full credit to the authors Adam Paszke and Mark Towers (tutorial in references)

training_step()

Description:
If we have enough experiences in the replay buffer, retrieve parameters.batch_size random experience samples
These are in the form of named tuples. We can convert these into an Experience object with one tuple for each named element
Experience((batch_size states), (batch_size actions), (batch_size rewards), (batch_size nextStates))

Further descriptions of these tuples and what they are used for are added as code comments.

-Calculate predicted values for actual action taken from current state using model
-Calculate actual values for nextState (having taken actual action)
-Calculate Mean Squared Error (MSE) over batch
-Compute gradients and backpropagate, updating model parameters
'''
def training_step():

    #if we don't have enough experiences in the replay buffer, skip the training step
    if replay_buffer.buffer_length() < hyperparameters.batch_size:
        return 

    experiences = replay_buffer.random_experience_batch()

    training_batch = Experience(*zip(*experiences))

    state_training_batch = torch.cat(training_batch.state) #observations passed through the model for Q(s, a))
    action_training_batch = torch.cat(training_batch.action) #used to store which action was actually chosen for state in each batch
    reward_training_batch = torch.cat(training_batch.reward) #used to calculate value (y) for both complete and non-complete nextStates
    nextState_training_batch = torch.cat([i for i in training_batch.nextState if i != None]) # for calculating actual values (only add non-complete nextStates to this training batch)

    #create bool mask highlighting indices of non-complete states
    non_complete_state_mask = torch.tensor(tuple(map(lambda s: s != None, training_batch.nextState)), dtype=torch.bool)


    state_vals = model.forward(state_training_batch)
    y_hat = torch.gather(state_vals, 1, action_training_batch) #predicted values for actual actions taken

    #create tensor filled with zeros, calculate actual values for non-complete nextState vals using this tensor with mask
    #don't calculate gradients here, no training
    nextState_vals = torch.zeros(hyperparameters.batch_size)

    with torch.no_grad():

        nextState_vals[non_complete_state_mask] = model.forward(nextState_training_batch).max(1).values

    #for complete nextStates we just take the reward, so 0 + reward for indices with complete nextState)
    y = (nextState_vals * hyperparameters.gamma) + reward_training_batch

    #use MSE as loss function, like in the original paper (y - y_hat)^2
    loss_function = torch.nn.MSELoss()
    loss_val = loss_function(y_hat, y.unsqueeze(1))

    #update model parameters
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()


'''
TRAINING LOOP
This is where we actually iterate over episodes and train our model.

Steps:

-Choose an action and record the nextState, reward, and game completion state after taking that action in the environment
-Decay epsilon
-Add the experience (state, action, reward, nextState) to the replay buffer
-Set current state to nextState. If current state is now complete (no observation), end the loop for the episode
-Train the model
-If the agent wins or loses the game, move to next episode
'''


#at start of every episode, reset environment and transform resulting starting observation to a Tensor object
for i in range(0, hyperparameters.num_episodes):

    done = False
    obs, info = env.reset()
    obs = transform_obs(obs)
    

    while done == False:

        #choose an action and update epsilon after decay (if not already 0.1)
        action = choose_action(obs, hyperparameters.epsilon)
        value_after_decay = hyperparameters.epsilon - (hyperparameters.epsilon * hyperparameters.epsilon_decay)

        if value_after_decay > 0.1:
            hyperparameters.epsilon = value_after_decay
        

        #take step using action in the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        reward = torch.tensor([reward])

        #get nextState and transform (if next state complete, set to None and do not transform)
        if terminated:
            next_obs = None

        else:
            next_obs = transform_obs(next_obs)

        #add experience to replay buffer
        replay_buffer.push(Experience(obs, action, reward, next_obs))

        #set current state to nextState- if complete, break loop (None isn't an image and can't be used as input for our model)!
        obs = next_obs

        if obs == None:
            break

        #train model
        training_step()

        #end loop if agent wins game or gets caught by an alien
        if terminated or truncated:
            done == True


print("Success")




    







    




