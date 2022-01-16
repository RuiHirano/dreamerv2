import gym
import dreamerv2.lib as dv2

config = dv2.defaults.update({
    'logdir': '~/logdir/breakout',
    'log_every': 1e3,
    'train_every': 10,
    'prefill': 1e5,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
}).parse_flags()
import gym
env = gym.make('Breakout-v0')
dreamerv2 = dv2.DreamerV2(env, config)
dreamerv2.train()