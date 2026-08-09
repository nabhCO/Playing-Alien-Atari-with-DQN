import gymnasium as gym
import torch
import torchvision
import numpy as np
from util import transform_obs
import parameters
from CNN_DQNAgent import ConvNeuralNet, ReplayBuffer, Experience
from gymnasium.wrappers import ResizeObservation, FrameStackObservation
from ale_py import ALEInterface

#SETUP

ale = ALEInterface()

#setting up environment and preprocessing- observation are image stacks of 4 that are grayscale, 84x84
env = gym.make('ALE/Alien-v5', obs_type="grayscale", render_mode="human") #change render_mode while training
env = ResizeObservation(env, (84, 84))
env = FrameStackObservation(env, 4) # return stack of most recent 4 observations

#initialize model, optimizer, replay buffer
model = ConvNeuralNet()
optimizer = torch.optim.RMSprop(model.parameters(), lr=parameters.alpha)
replay_buffer = ReplayBuffer(500)

#function to select action based on epsilon greedy strategy, decay epsilon value
def choose_action(obs, epsilon):

    choose_random_action = np.random.choice(a=[True, False], p=[epsilon, 1-epsilon])

    #apply linear epsilon decay as we will take a step with the action we choose
    new_epsilon = epsilon * parameters.epsilon_decay

    #with probability epsilon, we choose a random action to explore
    if choose_random_action:

        action = env.action_space.sample()

        #we're putting action in this form so it matches the model output (with an extra dimension for batch size)
        return torch.tensor([[action]], dtype=torch.long), new_epsilon

    #with probability 1 - epsilon, we use our model to decide the action for us
    #don't calculate gradients, we're not training the model right now
    with torch.no_grad():

        action_vals = model.forward(obs)
        return action_vals.max(1).indices.view(1, 1), new_epsilon #returns the index for max value action

    
def training_step():

    #if we don't have enough experiences in the replay buffer, skip the training step
    if replay_buffer.buffer_length() < parameters.batch_size:
        return 

    #collect 32 random experiences from the replay buffer
    experiences = replay_buffer.random_experience_batch()


    #convert the batch into an Experience object containing tuples of each of its components (e.g. tuple for state, tuple for action, etc.)
    training_batch = Experience(*zip(*experiences))

    #concatenates tensors contained in each of training batch's tuples
    state_training_batch = torch.cat(training_batch.state)
    action_training_batch = torch.cat(training_batch.action)
    reward_training_batch = torch.cat(training_batch.reward)

    #only add non-complete nextStates to this training batch
    nextState_training_batch = torch.cat([i for i in training_batch.nextState if i != None])

    #create mask highlighting the non-complete states
    non_complete_state_mask = torch.tensor(tuple(map(lambda s: s != None, training_batch.nextState)), dtype=torch.bool)


    #calculate predicted values for actual actions taken in these sample batch states
    state_vals = model.forward(state_training_batch)
    y_hat = torch.gather(state_vals, 1, action_training_batch)

    #create tensor filled with zeros
    nextState_vals = torch.zeros(parameters.batch_size)

    #calculate actual values for non-complete nextState vals
    with torch.no_grad():

        nextState_vals[non_complete_state_mask] = model.forward(nextState_training_batch).max(1).values

    #Q-value calculation (for complete nextStates we just take the reward, so 0 + reward for indices with complete nextState)
    y = (nextState_vals * parameters.gamma) + reward_training_batch

    #use MSE as loss function, like in the original paper (y - y_hat)^2
    loss_function = torch.nn.MSELoss()
    loss_val = loss_function(y_hat, y.unsqueeze(1))

    #update parameters
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()


#TRAINING

'''ADD COMMENTS PLS'''

for i in range(0, parameters.num_episodes):

    done = False

    obs, info = env.reset()

    obs = transform_obs(obs)
    

    while done == False:

        action, new_epsilon = choose_action(obs, parameters.epsilon)

        parameters.epsilon = new_epsilon

        next_obs, reward, terminated, truncated, info = env.step(action)
        reward = torch.tensor([reward])

        if terminated:

            next_obs = None

        else:

            next_obs = transform_obs(next_obs)

        replay_buffer.push(Experience(obs, action, reward, next_obs))

        obs = next_obs

        if obs == None:

            break

        training_step()

        if terminated or truncated:

            done == True


print("Success")




    







    




