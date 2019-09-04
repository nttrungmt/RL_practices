import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import argparse
import sys

import dqn
from dqn_utils import *
from atari_wrappers import *


def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def atari_learn(env,
                session,
                num_timesteps,
                log_file = './logs_pkls/rewards.pkl'):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10,
        log_file=log_file
    )
    env.close()


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)


def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)

    # This was the default provided in the starter code.
    #session = tf.Session(config=tf_config)

    # Use this if I want to see what is on the GPU.
    #session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # Use this for limiting memory allocated for the GPU.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


def get_env(task, seed):
    env_id = task.env_id
    env = gym.make(env_id)
    set_global_seeds(seed)
    env.seed(seed)
    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)
    return env


def main():
    # Games that we'll be testing.
    game_to_ID = {'BeamRider':0,
                  'Breakout':1,
                  'Enduro':2,
                  'Pong':3,
                  'Qbert':4}

    # Get some arguments here. Note: num_timesteps default uses tasks default.
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='Pong')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_timesteps', type=int, default=40000000)
    args = parser.parse_args()

    # Choose the game to play and set log file.
    benchmark = gym.benchmark_spec('Atari40M')
    task = benchmark.tasks[game_to_ID[args.game]]
    log_name = args.game+"_s"+str(args.seed).zfill(3)+".pkl"

    # Run training. Should change the seed if possible!
    # Also, the actual # of iterations run is _roughly_ num_timesteps/4.
    seed = args.seed
    env = get_env(task, seed)
    session = get_session()
    print("task = {}".format(task))
    atari_learn(env, 
                session, 
                num_timesteps=args.num_timesteps,
                log_file=log_name)

if __name__ == "__main__":
    main()
