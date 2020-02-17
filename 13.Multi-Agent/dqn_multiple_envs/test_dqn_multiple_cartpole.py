import os
import sys
import math
import random
import argparse
from datetime import datetime, date
import multiprocessing
import numpy as np
import tensorflow as tf
from gym.spaces import Box
from gym.vector.tests.utils import make_env
from gym.vector.sync_vector_env import SyncVectorEnv
from DQNAgent import DQNAgent

def main():
    np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})
    env_fns = [make_env('MountainCar-v0', i) for i in range(4)]
    try:
        env = SyncVectorEnv(env_fns)
    finally:
        env.close()
    
    state_size = env.observation_space.shape[1]
    action_size = env.action_space[0].n
    
    NUM_EPISODES = 1000
    STEPS_PER_EPISODE = 200
    batch_size = 32
    eps_mean_reward = [0.0] * NUM_EPISODES
    
    agent = DQNAgent(state_size, action_size)
    start_time = datetime.now()
    for ep_count  in range(NUM_EPISODES):
        episode_rew = 0
        state = env.reset()
        if(ep_count==0):
            print("ep={} state.shape={}".format(ep_count, state.shape))
        #state = np.reshape(state, [-1, state_size])
        ep_start_time = datetime.now()
        for time in range(STEPS_PER_EPISODE):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            episode_rew += np.sum(reward)
            #next_state = np.reshape(next_state, [-1, state_size])
            if(time==0):
                print("ep={} time={} action.len={} next_state.shape={} elaps_time={}".format( \
                    ep_count, time, len(action), next_state.shape, (datetime.now() - ep_start_time)) )
            #add to DQN buffer
            for idx in range(0, env.num_envs):
                agent.memorize(state[idx], action[idx], reward[idx], next_state[idx], done[idx])
            state = next_state
            if time >= STEPS_PER_EPISODE-1:
                eps_mean_reward[ep_count] = np.mean(episode_rew)/time
                print("ep: {}/{}, mean_avg_reward: {}, exec_time= {}".format( \
                    ep_count , NUM_EPISODES, eps_mean_reward[ep_count], (datetime.now() - ep_start_time)))
            #update DQN model if there are enough samples
            if len(agent.memory) > batch_size and time%8 == 0:
                agent.replay(batch_size)
        #if ep_count % 2 == 0:
        #    agent.save(str(os.path.join(save_path,'ma-foraging-dqn.h5')))
    print("Finish train DQN Agent with {} episodes in {}".format(NUM_EPISODES, (datetime.now() - start_time)))

if __name__ == '__main__':
    main()