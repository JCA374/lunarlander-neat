"""Flying one LunarLander episode with a NEAT network."""

import gymnasium as gym
import neat
import numpy as np

ENV_ID = 'LunarLander-v2'
MAX_STEPS = 1000
# Render geometry of the 600x400 rgb_array frame (30 px per metre), used to draw flight paths
VIEW_W, VIEW_H, SCALE = 600, 400, 30.0


def make_env(render_mode=None):
    return gym.make(ENV_ID, render_mode=render_mode)


def make_net(genome, config):
    return neat.nn.FeedForwardNetwork.create(genome, config)


def choose_action(net, obs):
    """The network outputs one score per action; the highest score fires."""
    return int(np.argmax(net.activate(obs)))


def run_episode(net, seed, env=None, frames=False, path=False, trace=False):
    """Fly one episode. Returns a dict with reward, steps, outcome and optional recordings:

    frames: rgb arrays, path: lander positions in frame pixels,
    trace: per-step observation, action and every node's value (for drawing a live network).
    """
    own_env = env is None
    if own_env:
        env = make_env('rgb_array' if frames else None)
    obs, _ = env.reset(seed=int(seed))
    out = {'seed': int(seed), 'reward': 0.0, 'frames': [], 'path': [], 'states': [],
           'actions': [], 'values': []}
    terminated = truncated = False
    reward = 0.0
    for _ in range(MAX_STEPS):
        if frames:
            out['frames'].append(env.render())
        if path:
            out['path'].append(_lander_px(env))
        action = choose_action(net, obs)
        if trace:
            out['states'].append(np.array(obs))
            out['actions'].append(action)
            out['values'].append(dict(net.values))
        obs, reward, terminated, truncated, _ = env.step(action)
        out['reward'] += float(reward)
        out['steps'] = out.get('steps', 0) + 1
        if terminated or truncated:
            break
    if frames:
        out['frames'].append(env.render())
    if path:
        out['path'].append(_lander_px(env))
    if own_env:
        env.close()
    # The env ends with +100 when the lander comes to rest and -100 on a crash or leaving the screen
    if terminated and reward > 0:
        out['outcome'] = 'landed'
    elif terminated:
        out['outcome'] = 'crashed'
    else:
        out['outcome'] = 'timeout'
    out['landed'] = out['outcome'] == 'landed'
    return out


def _lander_px(env):
    p = env.unwrapped.lander.position
    return (p.x * SCALE, VIEW_H - p.y * SCALE)
