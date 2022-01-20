import gym
import dreamerv2.lib as dv2
import datetime
import io
import uuid
import numpy as np
from pathlib import Path
  
env = gym.make("Pong-v0")
config = dv2.defaults.update({
    'task': 'atari_pong',
    'logdir': '~/MyProjects/FelixPort/data_science/imitaion/expert/pong/dreamerv2/1',
    'log_every': 1e3,
    'train_every': 10,
    'prefill': 1e5,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
}).parse_flags()

def save_episode(directory, episode):
    length = len(episode['action']) - 1
    ep = {'state': [], 'next_state': [], 'done': [], 'reward': [], 'action': []}
    ep['state'].extend(episode['image'][:length])
    ep['next_state'].extend(episode['image'][1:])
    ep['done'].extend(episode['is_terminal'][1:])
    ep['reward'].extend(episode['reward'][1:])
    ep['action'].extend(episode['reward'][1:])
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4().hex)
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **ep)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())
    return filename

agent = dv2.DreamerV2Agent(env, config, 'atari')
demo_dir = Path(__file__).parent.joinpath("demo").resolve()
def per_episode(ep):
    save_episode(demo_dir, ep)
agent.on_episode(per_episode)
agent.run_driver(num_episodes=1000)
