from __future__ import division, print_function

import time

import numpy as np
# import theano
# import theano.tensor as T
from base import ServoingOptimizationAlgorithm

from rl_util import do_rollouts, split_observations

from time_util import tic, toc


class ServoingFittedQIterationAlgorithm(ServoingOptimizationAlgorithm):
	def __init__(self, env, servoing_pol, sampling_iters, algorithm_iters,
				 num_trajs=None, num_steps=None, gamma=None, act_std=None,
				 iter_=0, thetas=None, mean_returns=None, std_returns=None,
				 mean_discounted_returns=None, std_discounted_returns=None,
				 learning_values=None, snapshot_interval=1, snapshot_prefix='',
				 plot=True, skip_validation=False, l2_reg=0.0, max_batch_size=10000000, max_memory_size=0,
				 eps=None, fit_alpha_bias=True, opt_fit_bias=False):
		super(ServoingFittedQIterationAlgorithm, self).__init__(env, servoing_pol, sampling_iters,
																num_trajs=num_trajs, num_steps=num_steps,
																gamma=gamma, act_std=act_std,
																iter_=iter_, thetas=thetas,
																mean_returns=mean_returns,
																std_returns=std_returns,
																mean_discounted_returns=mean_discounted_returns,
																std_discounted_returns=std_discounted_returns,
																learning_values=learning_values,
																snapshot_interval=snapshot_interval,
																snapshot_prefix=snapshot_prefix,
																plot=plot, skip_validation=skip_validation)
		self.algorithm_iters = algorithm_iters
		self.l2_reg = l2_reg
		self.max_batch_size = max_batch_size
		self.max_memory_size = max_memory_size
		self.eps = eps
		self.fit_alpha_bias = fit_alpha_bias
		self.opt_fit_bias = opt_fit_bias
		self.memory_sars = [], [], [], []
		self._bias = 0.0

	@property
	def theta(self):
		return self.servoing_pol.theta

	@theta.setter
	def theta(self, theta):
		self.servoing_pol.theta[...] = theta

	@property
	def bias(self):
		return self._bias

	@bias.setter
	def bias(self, bias):
		self._bias = float(bias)

	def iteration(self):
		_, observations, actions, rewards = do_rollouts(self.env, self.noisy_pol, self.num_trajs, self.num_steps,
														seeds=np.arange(self.num_trajs))
		def preprocess_obs(obs):
			for obs_name, obs_ in obs.items():
				if obs_name.endswith('image'):
					obs[obs_name], = self.servoing_pol.predictor.preprocess([obs_])
			return obs
		# preprocess_action = self.servoing_pol.predictor.transformers['action'].preprocess
		# observations = [[preprocess_obs(obs) for obs in observations_] for observations_ in observations]
		# actions = [[preprocess_action(action) for action in actions_] for actions_ in actions]
		rewards = [[-reward for reward in rewards_] for rewards_ in rewards]  # use costs
		observations, observations_p = split_observations(observations)

		return self.update(observations, actions, rewards, observations_p)

	def update(self, *sars):
		assert len(sars) == 4
		sars = [[step_data for traj_data in data for step_data in traj_data] for data in sars]

		orig_batch_size = len(sars[0])
		for data in sars[1:]:
			assert len(data) == orig_batch_size
		batch_size = min(orig_batch_size, self.max_batch_size)

		bellman_errors = []
		proxy_bellman_errors = []
		iter_ = 0
		while iter_ <= self.algorithm_iters:
			# if orig_batch_size > self.max_batch_size:
			# 	choice = np.random.choice(orig_batch_size, self.max_batch_size)
			# 	S, A, R, S_p = [[data[i] for i in choice] for data in sars]
			# elif iter_ == 0:  # the data is fixed across iterations so only set at the first iteration
			S, A, R, S_p = sars			

			# compute phi only if (S, A) has changed
			if orig_batch_size > self.max_batch_size or iter_ == 0:
				assert len(S) == batch_size
				A = np.asarray(A)
				R = np.asarray(R)
				tic()
				phi = self.servoing_pol.phi(S, A, preprocessed=True)
				# print(phi)
				toc("\tphi")

			# compute Q_sample
			tic()
			if self.gamma == 0:
				Q_sample = R
			else:
				# if orig_batch_size > self.max_batch_size:
				# 	A_p = self.servoing_pol.pi(S_p, preprocessed=True)
				# 	phi_p = self.servoing_pol.phi(S_p, A_p, preprocessed=True)
				# else:
				# 	if iter_ == 0:  # S_p is fixed across iterations so only compute A_b_c_split values at the first iteration
				# 		tic()
				# 		A_split, b_split, c_split = self.servoing_pol.A_b_c_split(S_p, preprocessed=True)
				# 		toc("\tA_b_c_split")
				# 	A_p = np.linalg.solve(
				# 		np.tensordot(A_split, self.servoing_pol.w / self.servoing_pol.repeats, axes=(0, 0)) + np.diag(self.servoing_pol.lambda_),
				# 		np.tensordot(b_split, self.servoing_pol.w / self.servoing_pol.repeats, axes=(0, 0))
				# 	)
				# 	A_p = np.array([self.servoing_pol.action_transformer.deprocess(a_p) for a_p in A_p])
				# 	for a_p in A_p:
				# 		self.servoing_pol.action_space.clip(a_p, out=a_p)
				# 	A_p = np.array([self.servoing_pol.action_transformer.preprocess(a_p) for a_p in A_p])
				# 	phi_p_errors = np.einsum('injk,nk,nj->ni', A_split, A_p, A_p) - 2 * np.einsum('inj,nj->ni', b_split, A_p) + c_split.T
				# 	phi_p_actions = A_p ** 2
				# 	phi_p = np.concatenate([phi_p_errors / self.servoing_pol.repeats, phi_p_actions], axis=1)
				if iter_ == 0:
					A_p = self.servoing_pol.pi(S_p, preprocessed=True)
					phi_p = self.servoing_pol.phi(S_p, A_p, preprocessed=True)
				V_p = phi_p.dot(self.theta) + self.bias
				Q_sample = R + self.gamma * V_p
			toc("\tQ_sample")

			if self.fit_alpha_bias:
				old_objective_value = (1 / 2.) * ((phi.dot(self.theta) + self.bias - Q_sample) ** 2).mean() + \
									  (self.l2_reg / 2.) * (self.theta ** 2).sum()
				L = np.c_[phi.dot(self.theta) - self.gamma * phi_p.dot(self.theta), (1 - self.gamma) * np.ones(batch_size)]
				D = np.diag([batch_size * self.l2_reg * (self.theta ** 2).sum(), 0])

				lsq_A_fit = L.T.dot(L) + D
				lsq_b_fit = L.T.dot(R)
				alpha, bias = np.linalg.solve(lsq_A_fit, lsq_b_fit)
				if alpha <= 0:
					print("\tUnconstrained alpha is negative (alpha = %.2f). Solving constrained optimization." % alpha)
					import cvxpy
					x_var = cvxpy.Variable(2)
					objective = cvxpy.Minimize((1 / 2.) * cvxpy.sum_squares(lsq_A_fit * x_var - lsq_b_fit))
					constraints = [0.0 <= x_var[0]]
					prob = cvxpy.Problem(objective, constraints)
					solved = False
					for solver in [None, cvxpy.GUROBI, cvxpy.CVXOPT]:
						try:
							prob.solve(solver=solver)
						except cvxpy.error.SolverError:
							continue
						if x_var.value is None:
							continue
						solved = True
						break
					if solved:
						alpha, bias = np.squeeze(np.array(x_var.value))
					else:
						print("\tUnable to solve constrained optimization. Setting alpha = 0, solving for bias.")
						alpha = 0.0
						bias = R.mean() / (1 - self.gamma)

				# print("Alpha " + str(alpha))
				self.theta *= alpha
				self.bias = bias
				if self.gamma == 0:
					Q_sample = R
				else:
					# the policy shouldn't have changed (because alpha >= 0), so phi_p doesn't need to be updated
					V_p = phi_p.dot(self.theta) + self.bias
					Q_sample = R + self.gamma * V_p
				new_objective_value = (1 / 2.) * ((phi.dot(self.theta) + self.bias - Q_sample) ** 2).mean() + \
									  (self.l2_reg / 2.) * (self.theta ** 2).sum()
				if new_objective_value > old_objective_value:
					print("\tObjective increased from %.6f to %.6f after fitting alpha and bias." % (old_objective_value, new_objective_value))

			objective_value = (1 / 2.) * ((phi.dot(self.theta) + self.bias - Q_sample) ** 2).mean() + \
							  (self.l2_reg / 2.) * (self.theta ** 2).sum()
			print("\tIteration {} of {}".format(iter_, self.algorithm_iters))
			print("\t    bellman error = {:.6f}".format(objective_value))
			bellman_errors.append(objective_value)

			if iter_ < self.algorithm_iters:  # don't take an update step after the last iteration
				proxy_bellman_error = self.fqi_update(S, A, R, S_p, phi=phi, Q_sample=Q_sample)
				objective_increased = proxy_bellman_error > objective_value
				print(u"\t                  {} {:.6f}".format(u"\u2191" if objective_increased else u"\u2193", proxy_bellman_error))
				proxy_bellman_errors.append(proxy_bellman_error)

			iter_ += 1

		return bellman_errors

	def fqi_update(self, S, A, R, S_p, phi=None, Q_sample=None):
		batch_size = len(S)
		if phi is None:
			tic()
			A = np.asarray(A)
			R = np.asarray(R)
			tic()
			phi = self.servoing_pol.phi(S, A, preprocessed=True)
			toc("\tphi")

		if Q_sample is None:
			tic()
			if self.gamma == 0:
				Q_sample = R
			else:
				A_p = self.servoing_pol.pi(S_p, preprocessed=True)
				phi_p = self.servoing_pol.phi(S_p, A_p, preprocessed=True)
				V_p = phi_p.dot(self.theta) + self.bias
				Q_sample = R + self.gamma * V_p
			toc("\tQ_sample")

		tic()
		import cvxpy
		theta_var = cvxpy.Variable(self.theta.shape[0])
		bias_var = cvxpy.Variable(1)
		assert len(phi) == batch_size
		scale = 1.0
		solved = False
		while not solved:
			if self.opt_fit_bias:
				objective = cvxpy.Minimize(
					(1 / 2.) * cvxpy.sum_squares(
						(phi * np.sqrt(scale / batch_size)) * theta_var + bias_var * np.sqrt(scale / batch_size) -
						((Q_sample + (bias_var - self.bias) * self.gamma) * np.sqrt(scale / batch_size))) +
					(self.l2_reg / 2.) * scale * cvxpy.sum_squares(theta_var))  # no regularization on bias
			else:
				objective = cvxpy.Minimize(
					(1 / 2.) * cvxpy.sum_squares(
						(phi * np.sqrt(scale / batch_size)) * theta_var + bias_var * np.sqrt(scale / batch_size) -
						(Q_sample * np.sqrt(scale / batch_size))) +
					(self.l2_reg / 2.) * scale * cvxpy.sum_squares(theta_var))  # no regularization on bias
			constraints = [0 <= theta_var]  # no constraint on bias

			if self.eps is not None:
				constraints.append(cvxpy.sum_squares(theta_var - self.theta) <= (len(self.theta) * self.eps))

			prob = cvxpy.Problem(objective, constraints)
			for solver in [None, cvxpy.GUROBI, cvxpy.CVXOPT]:
				try:
					prob.solve(solver=solver)
				except cvxpy.error.SolverError:
					continue
				if theta_var.value is None:
					continue
				solved = True
				break

			if not solved:
				scale *= 0.1
				print("Unable to solve FQI optimization. Reformulating objective with a scale of %f." % scale)
		# self.theta = np.squeeze(np.array(theta_var .value), axis=1)
		self.theta = np.array(theta_var.value)
		self.bias = bias_var.value
		proxy_bellman_error = objective.value / scale
		toc("\tcvxpy")
		print(self.theta)
		print(self.bias)
		return proxy_bellman_error