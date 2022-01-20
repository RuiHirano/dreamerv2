import collections
import logging
import os
import pathlib
import re
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import agent
import common

from common import Config
from common import GymWrapper
from common import RenderImage
from common import TerminalOutput
from common import JSONLOutput
from common import TensorBoardOutput

configs = yaml.safe_load(
    (pathlib.Path(__file__).parent / 'configs.yaml').read_text())
defaults = common.Config(configs.pop('defaults'))

class DreamerV2:
	def __init__(self, env, config):
		self._env = self._create_env(env, config)
		self._config = config
		self._on_per_train_epoch_funcs = []
		self._episodes_per_train_epoch = []

		self._logdir = pathlib.Path(self._config.logdir).expanduser()
		self._logdir.mkdir(parents=True, exist_ok=True)
		self._config.save(self._logdir / 'config.yaml')
		outputs = [
			common.TerminalOutput(),
      		common.JSONLOutput(self._config.logdir),
    		common.TensorBoardOutput(self._config.logdir),
		]
		self._replay = common.Replay(self._logdir / 'train_episodes', **self._config.replay)
		self._step = common.Counter(self._replay.stats['total_steps'])
		self._logger = common.Logger(self._step, outputs, multiplier=self._config.action_repeat)
		self._metrics = collections.defaultdict(list)
		self._driver = common.Driver([self._env])
		self._driver.on_episode(self._on_per_episode)
		self._driver.on_step(lambda tran, worker: self._step.increment())
		self._driver.on_step(self._replay.add_step)
		self._driver.on_reset(self._replay.add_step)

		self._should_log = common.Every(self._config.log_every)
		self._should_video = common.Every(self._config.log_every)
		self._should_expl = common.Until(self._config.expl_until)

		# 1. prefill dataset
		self._prefill()
		# 2. create agent and prepare training
		self._create_agent()
	
	def on_per_train_epoch(self, func):
		self._on_per_train_epoch_funcs.append(func)

	def get_replay_episodes(self):
		return self._replay._complete_eps

	def _create_env(self, env, config):
		env = common.GymWrapper(env)
		env = common.ResizeImage(env)
		if hasattr(env.act_space['action'], 'n'):
			env = common.OneHotAction(env)
		else:
			env = common.NormalizeAction(env)
		env = common.TimeLimit(env, config.time_limit)
		return env

	def _create_agent(self):
		self._agent = agent.Agent(self._config, self._env.obs_space, self._env.act_space, self._step)
		self._dataset = iter(self._replay.dataset(**self._config.dataset))
		self._train_agent = common.CarryOverState(self._agent.train)
		self._train_agent(next(self._dataset))
		if (self._logdir / 'variables.pkl').exists():
			self._agent.load(self._logdir / 'variables.pkl')
		else:
			print('Pretrain agent.')
			for _ in range(self._config.pretrain):
				self._train_agent(next(self._dataset))
		self._policy = lambda *args: self._agent.policy(
    		*args, mode='explore' if self._should_expl(self._step) else 'train')

	def _prefill(self):
		prefill = max(0, self._config.prefill - self._replay.stats['total_steps'])
		if prefill:
			print(f'Prefill dataset ({prefill} steps).')
			random_agent = common.RandomAgent(self._env.act_space)
			self._driver(random_agent, steps=prefill, episodes=1)
			self._driver.reset()

	def _on_per_episode(self, ep):
		self._replay.add_episode(ep)
		length = len(ep['reward']) - 1
		score = float(ep['reward'].astype(np.float64).sum())
		print(f'Episode has {length} steps and return {score:.1f}.')
		self._logger.scalar('return', score)
		self._logger.scalar('length', length)
		for key, value in ep.items():
			if re.match(self._config.log_keys_sum, key):
				self._logger.scalar(f'sum_{key}', ep[key].sum())
			if re.match(self._config.log_keys_mean, key):
				self._logger.scalar(f'mean_{key}', ep[key].mean())
			if re.match(self._config.log_keys_max, key):
				self._logger.scalar(f'max_{key}', ep[key].max(0).mean())
		if self._should_video(self._step):
			for key in self._config.log_keys_video:
				self._logger.video(f'policy_{key}', ep[key])
		self._logger.add(self._replay.stats)
		self._logger.write()
	
	def _on_per_train_episode(self, ep):
		self._episodes_per_train_epoch.append(ep)
		if self._should_log(self._step):
			for name, values in self._metrics.items():
				print(name, values)
				self._logger.scalar(name, np.array(values, np.float64).mean())
				self._metrics[name].clear()
			self._logger.add(self._agent.report(next(self._dataset)))
			self._logger.write(fps=True)

	def train(self):
		self._driver.on_episode(self._on_per_train_episode)
		self._episodes_per_train_epoch = []
		while self._step < self._config.steps:
			# rollout
			self._driver(self._policy, episodes=self._config.rollout_episodes)
			# train
			for _ in range(self._config.train_steps):
				mets = self._train_agent(next(self._dataset))
				[self._metrics[key].append(value) for key, value in mets.items()]
			# save policy
			self._agent.save(self._logdir / 'variables.pkl')
			# callback
			[func(self._episodes_per_train_epoch) for func in self._on_per_train_epoch_funcs]

class DreamerV2Agent:
	def __init__(self, env, config, type="atari"):
		config = config.update(configs['atari'])

		logdir = pathlib.Path(config.logdir).expanduser()

		replay = common.Replay(logdir / 'eval_episodes', **config.replay)
		step = common.Counter(replay.stats['total_steps'])

		def make_env():
			suite, task = "atari_pong".split('_', 1)
			env = common.Atari(
				task, config.action_repeat, config.render_size,
				config.atari_grayscale)
			env = common.OneHotAction(env)
			env = common.TimeLimit(env, config.time_limit)
			return env
		self._env = make_env()
		'''env = common.GymWrapper(env)
		env = common.ResizeImage(env)
		if hasattr(env.act_space['action'], 'n'):
			env = common.OneHotAction(env)
		else:
			env = common.NormalizeAction(env)
		env = common.TimeLimit(env, config.time_limit)'''

		print('Create agent.')
		self.agnt = agent.Agent(config, self._env.obs_space, self._env.act_space, step)
		dataset = iter(replay.dataset(**config.dataset))
		train_agent = common.CarryOverState(self.agnt.train)
		train_agent(next(dataset))
		self.agnt.load(logdir / "variables.pkl")
		self._state = None
		self._ob = None
		self._result = {"step": 0, "total_reward": 0}
		self._driver = common.Driver([self._env])

	def on_episode(self, fn):
		self._driver.on_episode(fn)

	def run_driver(self, num_episodes):
		self._driver(self.agnt.policy, episodes=num_episodes)

if __name__ == '__main__':
	config = defaults.update({
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
	dv2 = DreamerV2(env, config)
	dv2.train()