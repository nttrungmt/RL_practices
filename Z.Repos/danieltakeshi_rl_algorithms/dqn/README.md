# Deep Q-Networks

The starter code is from UC Berkeley's Deep Reinforcement Learning class.  These
are their comments:

> See http://rll.berkeley.edu/deeprlcourse/docs/hw3.pdf for instructions
> 
> The starter code was based on an implementation of Q-learning for Atari
> generously provided by Szymon Sidor from OpenAI

The rest of this README contains my comments and results. 

# Usage, Games, etc.

First, here is example usage (slashes are only for readability here):

```
python run_dqn_atari.py --game Pong --seed 1 --num_timesteps 30000000 | tee logs_text/Pong_s001.text
```

With these settings, the statistics for plotting data will be stored in the
`log_pkls/Pong_s001.pkl` file.

Here are some of the `task` stuff in the code, ordered by index (i.e. 0, 1,
etc.).

```
Task<env_id=BeamRiderNoFrameskip-v3 trials=2 max_timesteps=40000000 max_seconds=None reward_floor=363.9 reward_ceiling=60000.0>
Task<env_id=BreakoutNoFrameskip-v3 trials=2 max_timesteps=40000000 max_seconds=None reward_floor=1.7 reward_ceiling=800.0>
Task<env_id=EnduroNoFrameskip-v3 trials=2 max_timesteps=40000000 max_seconds=None reward_floor=0.0 reward_ceiling=5000.0>
Task<env_id=PongNoFrameskip-v3 trials=2 max_timesteps=40000000 max_seconds=None reward_floor=-20.7 reward_ceiling=21.0>
Task<env_id=QbertNoFrameskip-v3 trials=2 max_timesteps=40000000 max_seconds=None reward_floor=163.9 reward_ceiling=40000.0>
```

The default for these is 40 million episodes, but that's not always needed for
the easier games.

The `num_timesteps` parameter corresponds to the number of steps in the
"underlying" environment, *not* the "wrapped" environment. See the stopping
criterion:

```
def stopping_criterion(env, t):
    # notice that here t is the number of steps of the wrapped env,
    # which is different from the number of steps in the underlying env
    return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps
```

The `t` here is what I think of as "the number of steps." It's confusing, I
know. There might be a better way to handle this. From now on, when I say
"steps", it refers to the `t`-like number, and *not* the `num_timesteps`
parameter.

# Results

Notes:

- Timing results are based on running with an NVIDIA Titan X with Pascal GPU.
- Scores per Episode indicate scores for every episode.
- Scores per Timestep indicate the score of the current episode at a given
  timestep; each episode requires some number of timesteps for the agent to
  complete it. Theres' a lot of them, so I take every 10,000.
- Blocks mean taking an interval of some size (100) and taking the mean.

## Pong

Commands:

```
python run_dqn_atari.py --game Pong --seed 1 --num_timesteps 30000000 | tee logs_text/Pong_s001.text
python run_dqn_atari.py --game Pong --seed 2 --num_timesteps 30000000 | tee logs_text/Pong_s002.text
```

- `num_timesteps`: 30 million
- Training steps: about 7.5 million
- Episodes: about 4200
- Time: about 9.0 hours

![pong](figures/Pong.png?raw=true)

## Breakout

Command:

```
python run_dqn_atari.py --game Breakout --seed 1 --num_timesteps 40000000 | tee logs_text/Breakout_s001.text
python run_dqn_atari.py --game Breakout --seed 2 --num_timesteps 40000000 | tee logs_text/Breakout_s002.text
```

- `num_timesteps`: 40 million
- Training steps: about 9.7 million
- Episodes: about 10,200
- Time: about 11.8 hours

I have no idea why the performance plummeted after 4500 episodes. I may need to
investigate. Note that a few times we get the absolute perfect highest score
(two full boards cleared); I think the "800" value as the maximum score is wrong
...

![breakout](figures/Breakout.png?raw=true)


## BeamRider

Command:

```
python run_dqn_atari.py --game BeamRider --seed 1 --num_timesteps 40000000 | tee logs_text/BeamRider_s001.text
python run_dqn_atari.py --game BeamRider --seed 2 --num_timesteps 40000000 | tee logs_text/BeamRider_s002.text
```

- `num_timesteps`: 40 million
- Training steps: about 10.0 million
- Episodes: about 2,500 to 4,000
- Time: about 12.5 hours

The results look different among the seeds since seed 2 apparently had better
peformance earlier, thus meaning its episodes became longer sooner than the seed
1 version. At least they look reasonably good. A3C got roughly 13k on this game.

![beamrider](figures/BeamRider.png?raw=true)


## Enduro

Command:

```
python run_dqn_atari.py --game Enduro --seed 1 --num_timesteps 40000000 | tee logs_text/Enduro_s001.text
python run_dqn_atari.py --game Enduro --seed 2 --num_timesteps 40000000 | tee logs_text/Enduro_s002.text
```

- `num_timesteps`: 40 million
- Training steps: about 10.0 million
- Episodes: about 2,500 to 3,000
- Time: about 12.5 hours

Ack, what happened to seed 2?!? The first seed matches the DQN result (475.6)
from the Nature script, yet I don't know why the second one failed to learn
much. It's worth noting, though, that the A3C paper actually reported -82.2 on
this game (really?!?). Oh well ...

![enduro](figures/Enduro.png?raw=true)
