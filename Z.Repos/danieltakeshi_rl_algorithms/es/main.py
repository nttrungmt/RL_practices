"""
Use this script for setting the arguments.

(c) May 2017 by Daniel Seita
"""

import argparse
import logz
import os
import pickle
import tensorflow as tf
import utils
from es import ESAgent


if __name__ == "__main__":
    """ LOTS of arguments here, but hopefully most are straightforward. Run
    `python main.py -h` to visualize the help messages.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str, 
            help='The OpenAI gym environment name (case sensitive).')
    parser.add_argument('--do_not_save', action='store_true',
            help='Sets the log_dir to be None.')
    parser.add_argument('--es_iters', type=int, default=100,
            help='Iterations to run ES.')
    parser.add_argument('--log_every_t_iter', type=int, default=1,
            help='Controls the amount of time information is logged.')
    parser.add_argument('--lrate_es', type=float, default=0.001,
            help='Learning rate for the ES gradient update.')
    parser.add_argument('--npop', type=int, default=200, 
            help='Weight vectors to sample for ES (INCLUDING the mirroring')
    parser.add_argument('--render', action='store_true',
            help='Use `--render` to visualize trajectories each iteration.')
    parser.add_argument('--seed', type=int, default=0,
            help='The random seed.')
    parser.add_argument('--sigma', type=float, default=0.1,
            help='Sigma (standard deviation) for the Gaussian noise.')
    parser.add_argument('--snapshot_every_t_iter', type=int, default=100,
            help='Save the model every t iterations so we can inspect later.')
    parser.add_argument('--test_trajs', type=int, default=10, 
            help='Number of evaluation trajectories after each iteration.')
    parser.add_argument('--verbose', action='store_true',
            help='Use `--verbose` for a few additional debugging messages.')
    args = parser.parse_args()
    assert args.npop % 2 == 0 # Just to be consistent with my other code.

    # Make the TensorFlow session and do some logic with handling arguments.
    session = utils.get_tf_session()
    log_dir = None
    if not args.do_not_save:
        log_dir = 'outputs/' +args.envname+ '/seed' +str(args.seed).zfill(4)
        logz.configure_output_dir(log_dir)
        os.makedirs(log_dir+'/snapshots/')
        with open(log_dir+'/args.pkl','wb') as f:
            pickle.dump(args, f)

    # Build and run evolution strategies.
    es_agent = ESAgent(session, args, log_dir)
    es_agent.run_es()
