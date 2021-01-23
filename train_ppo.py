import os
import torch
from torch.distributions import Normal
import numpy as np
import gfootball.env as football_env
import gym
from tqdm import tqdm

from models.actor import Actor
from models.critic import Critic
from models.ActorCritic import ActorCritic
from PPO import PPO

def test_env(env, model, device):
    state = env.reset()
    model.eval()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        state = state.permute(0, 3, 1, 2)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])

        state = next_state
        total_reward += reward
    print("Reward gained : ", total_reward)
    return total_reward


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # This is the main function
    env = football_env.create_environment(env_name='academy_empty_goal', stacked=False, rewards="scoring,checkpoints", representation='pixels', render=True)
    state = env.reset()

    state_dims = env.observation_space.shape
    print(state_dims)

    n_actions = env.action_space.n 
    print(n_actions)

    # Config options
    ppo_steps = 128
    ppo_epochs = 4
    batch_size = 5
    target_reached = False
    threshold_reward = 200
    max_iters = 50
    iters = 0

    model = ActorCritic(n_actions, 0.002, (0.9, 0.999))
    optimizer = model.optimizer
    model.to(device)
    
    ppo_updator = PPO(logdir='outputs/gfootball_512_set/') # currently lets not change anything

    optim = torch.optim.Adam(model.parameters(), lr=0.002)

    print(model)    
    # store the test rewards
    test_rewards = []

    while not target_reached and iters < max_iters:
        '''
        Iterate over all steps necessary
        '''
        states = []
        actions = []
        values = []
        masks = []
        rewards = []
        actions_probs = []
        entropy = 0

        for itr in tqdm(range(ppo_steps), desc=" Forward pass {} : ".format(iters)):
            # set model to train
            model.eval()

            # convert array state to tensor state
            state = torch.from_numpy(state).float().to(device)
            state = torch.unsqueeze(state, 0)
            state = state.permute(0, 3, 1, 2)
                        
            # jointly forward pass on actor and critic models
            action_dist, q_value = model(state)
                        
            # Perform an action
            action = action_dist.sample()
            next_state, reward, done, info = env.step(action.detach().cpu().numpy().squeeze())

            log_probs = action_dist.log_prob(action)
            entropy += ppo_updator._calc_entropy(action_dist)
                        
            #print(" Iter : {} State shape : {} Reward: {} Mask : {} ".format(itr, state.shape, reward, done))

            '''
            # NOTE : only if multiple envs are present
            reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
            mask = torch.FloatTensor(1 - done).unsqueeze(1).to(device)
            '''
            
            reward = torch.FloatTensor([reward]).to(device)
            mask = torch.FloatTensor([1 - done]).to(device)

            states.append(state)
            actions.append(action)
            actions_probs.append(log_probs)
            values.append(q_value)
            masks.append(mask)
            rewards.append(reward)

            state = next_state # update to next state
            if done:
                env.reset()
        # compute the actor critic prob distribution
        next_state = torch.from_numpy(next_state).float().to(device)
        next_state = torch.unsqueeze(next_state, 0)
        next_state = next_state.permute(0, 3, 1, 2)

        # compute advantage
        _, next_value = model(next_state) # this is the critic's decision
        returns =  ppo_updator.calc_returns(masks, values, rewards, next_value)
       
        returns = torch.cat(returns).detach()
        actions_probs = torch.cat([log_prob.unsqueeze(1) for log_prob in actions_probs]).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions).unsqueeze(1)
        advantages = returns - values
        
        # Compute the PPO loss and update model
        ppo_updator.update_backward(model, optimizer, states, actions, actions_probs, returns, advantages, iters)
        
        # Run tests
        if iters % 5 == 0:
            print("Running test ...")
            test_reward = np.mean([test_env(env, model, device) for _ in range(10)])
            test_rewards.append(test_reward)
            if test_reward > threshold_reward:
                target_reached = True
        
        # increment the iterator
        iters += 1 
        env.reset()

    env.close()
