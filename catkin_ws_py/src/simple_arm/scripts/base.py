import numpy as np
import yaml
from rl_util import do_rollouts, discount_returns
from additive_normal_policy import AdditiveNormalPolicy

class ServoingOptimizationAlgorithm(object):
	"""docstring for Algorithm"""
	def __init__(self, env, servoing_pol, sampling_iters, num_trajs=None, num_steps=None, gamma=None, act_std=None,
				 iter_=0, thetas=None, mean_returns=None, std_returns=None, mean_discounted_returns=None,
				 std_discounted_returns=None, learning_values=None, snapshot_interval=1, snapshot_prefix='',
				 plot=True, skip_validation=False):
		# super(Algorithm, self).__init__()
		# assert isinstance(env, envs.ServoingEnv)
  #       assert isinstance(servoing_pol, policies.ServoingPolicy)
		self.env = env
		self.sampling_iters = sampling_iters
		self.servoing_pol = servoing_pol
		self.num_trajs = 10 if num_trajs is None else num_trajs
		self.num_steps = 100 if num_steps is None else num_steps
		self.env.max_time_steps = self.num_steps
		self.gamma = 0.9 if gamma is None else gamma
		self.act_std = 0.4 if act_std is None else act_std
		# self.noisy_pol = AdditiveNormalPolicy(servoing_pol, env.action_space, None, act_std=self.act_std)
		self.noisy_pol = AdditiveNormalPolicy(servoing_pol, None, None, act_std=self.act_std)

		self.iter_ = iter_
		self.thetas = [np.asarray(theta) for theta in thetas] if thetas is not None else []
		self.mean_returns = [np.asarray(ret) for ret in mean_returns] if mean_returns is not None else []
		self.std_returns = [np.asarray(ret) for ret in std_returns] if std_returns is not None else []
		self.mean_discounted_returns = [np.asarray(ret) for ret in mean_discounted_returns] if mean_discounted_returns is not None else []
		self.std_discounted_returns = [np.asarray(ret) for ret in std_discounted_returns] if std_discounted_returns is not None else []
		self.learning_values = [np.asarray(value) for value in learning_values] if learning_values is not None else []
		# self.snapshot_interval = snapshot_interval
		# self.snapshot_prefix = snapshot_prefix
		# self.plot = plot
		self.skip_validation = skip_validation
		

	def run(self):

		while self.iter_ <= self.sampling_iters:
			print("Iteration {} of {}".format(self.iter_, self.sampling_iters))
			# self.thetas.append(self.servoing_pol.theta.copy())
			if not self.skip_validation:
				rewards = do_rollouts(self.env, self.servoing_pol, self.num_trajs, self.num_steps,
									  seeds=np.arange(self.num_trajs), ret_rewards_only=True)
				returns = discount_returns(rewards, 1.0)
				mean_return = np.mean(returns)
				std_return = np.std(returns) / np.sqrt(len(returns))
				self.mean_returns.append(mean_return)
				self.std_returns.append(std_return)
				discounted_returns = discount_returns(rewards, self.gamma)
				mean_discounted_return = np.mean(discounted_returns)
				std_discounted_return = np.std(discounted_returns) / np.sqrt(len(discounted_returns))
				self.mean_discounted_returns.append(mean_discounted_return)
				self.std_discounted_returns.append(std_discounted_return)
				print("    mean discounted return = {:.6f} ({:.6f})".format(mean_discounted_return, std_discounted_return))
				print("    mean return = {:.6f} ({:.6f})".format(mean_return, std_return))

			if self.iter_ < self.sampling_iters:
				learning_value = self.iteration()
				self.learning_values.append(np.asarray(learning_value))

			self.iter_ += 1

			