import os, math, argparse
import brain
import agent
import torch
from torch import multiprocessing as mp
from core import *

def run_agent(shared_brain, render=False, verbose=False, args=None):
    """
    Run the agent.

    Parameters
    ----------
    shared_brain : brain.Brain
        The shared brain the agents will use and update.
    render : boolean, optional
        Should the agent render its actions in the on-policy phase?
    verbose : boolean, optional
        Should the agent print progress to the console?
    """
    if CONTROL is 'discrete':
        local_agent = agent.DiscreteAgent(shared_brain, render, verbose, args)
    else:
        local_agent = agent.ContinuousAgent(shared_brain, render, verbose, args)
    local_agent.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ACER')
    parser.add_argument('--policy', type=str, default='RobotPolicy',help='Policy to train')
    parser.add_argument('--env-config-path', type=str, default='./configurations/2way_8_config.json',help='environment config file (JSON)')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--total-timesteps', type=int, default=64000, metavar='STEPS', help='Number of training steps')
    parser.add_argument('--nsteps', type=int, default=128, metavar='STEPS', help='number of steps of the vectorized environment per update')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='policy entropy coefficient in the optimization objective')
    parser.add_argument('--lr', type=float, default=0.0007, help='learning rate, constant or a schedule function')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='value function loss coefficient in the optimization objective')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='gradient norm clipping coefficient')
    parser.add_argument('--gamma', type=float, default=0.99, help='discounting factor')
    parser.add_argument('--lam', type=float, default=0.95, help='advantage estimation discounting factor (lambda in the paper)')
    parser.add_argument('--nminibatches', type=int, default=1, metavar='SIZE', help='number of training minibatches per update. <= number of environments run in parallel.')
    parser.add_argument('--noptepochs', type=int, default=10, metavar='SIZE', help='number of training epochs per update')
    parser.add_argument('--cliprange', type=float, default=0.3, help='clipping range, constant or schedule function [0,1]')
    parser.add_argument('--logdir', type=str, default='./log/acer/', help='Save folder for log, final model, tensor board')
    parser.add_argument('--log-interval', type=int, default=1, metavar='STEPS', help='number of timesteps between logging events')
    parser.add_argument('--save-interval', type=int, default=10, metavar='SIZE', help='number of timesteps between saving events')
    parser.add_argument('--restore-path', type=str, default=None, help='Path to restore previously saved model')
    parser.add_argument('--last-timesteps', type=int, default=0, metavar='STEPS', help='Last training time steps')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only, no training, must provided saved model')
    parser.add_argument('--evaluate-episodes', type=int, default=10, metavar='EPISODES', help='Number of evaluation episodes')
    #parser.add_argument('--shuffle-goal', action='store_true', help='shuffle goal poses during training')
    #parser.add_argument('--enable-reset-early', action='store_true', help='can reset early if all robots collide')
    parser.add_argument('--mdl-name-prefix', type=str, default='ppo',help='prefix of model name to save')
    
    args = parser.parse_args()
    
    if NUMBER_OF_AGENTS == 1:
        # Don't bother with multiprocessing if only one agent
        run_agent(brain.brain, render=True, args=args)
    else:
        processes = [mp.Process(target=run_agent, args=(brain.brain, False, True, args))
                     for a_i in range(NUMBER_OF_AGENTS)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        
        save_path = str(os.path.join(args.logdir,ENVIRONMENT_NAME))
        print('Log dir will be configured at {}'.format(save_path))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        brain.brain.save(save_path)

def test():
    run_agent(brain.brain, render=True)
