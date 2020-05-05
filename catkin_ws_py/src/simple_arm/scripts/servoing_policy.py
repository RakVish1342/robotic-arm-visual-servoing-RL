from __future__ import division, print_function

import time

# import lasagne.layers as L
import numpy as np
import theano
import theano.tensor as T
import pickle
# import theano
# import theano.tensor as T
# import yaml


class ServoingPolicy():
	def __init__(self, alpha=1.0, lambda_=0.0, w=1.0, use_constrained_opt=False, unweighted_features=False, algorithm_or_fname=None):
		# if isinstance(predictor, str):
		#     with open(predictor) as predictor_file:
		#         predictor = from_yaml(predictor_file)
		# self.predictor = predictor
		# self.action_transformer = self.predictor.transformers['u']
		# self.action_space = from_config(self.predictor.environment_config['action_space'])
		self.alpha = alpha
		lambda_ = np.asarray(lambda_)
		if np.isscalar(lambda_) or lambda_.ndim == 0:
			# lambda_ = lambda_ * np.ones(self.action_space.shape)  # numpy fails with augmented assigment
			lambda_ = lambda_ * np.ones(2)
		# assert lambda_.shape == self.action_space.shape
		self._lambda_ = lambda_
		# feature_names = iter_util.flatten_tree(self.predictor.feature_name)
		# feature_shapes = L.get_output_shape([self.predictor.pred_layers[name] for name in feature_names])
		self.repeats = [1]
		# for feature_shape in feature_shapes:
		#     self.repeats.extend([np.prod(feature_shape[2:])] * feature_shape[1])
		w = np.asarray(w)
		if np.isscalar(w) or w.ndim == 0 or len(w) == 1:
			w = w * np.ones(len(self.repeats))  # numpy fails with augmented assigment
		# elif w.shape == (len(feature_names),):
		#     w = np.repeat(w, [feature_shape[1] for feature_shape in feature_shapes])
		assert w.shape == (len(self.repeats),)
		self._w = w
		self._theta = np.append(self._w, self._lambda_)
		self._w, self._lambda_ = np.split(self._theta, [len(self._w)])  # alias the parameters
		self.use_constrained_opt = use_constrained_opt
		self.unweighted_features = unweighted_features
		# self.image_name = 'image'
		# self.target_image_name = 'target_image'

		# if algorithm_or_fname is not None:
		#     from visual_dynamics.algorithms import ServoingFittedQIterationAlgorithm
		#     if isinstance(algorithm_or_fname, str):
		#         with open(algorithm_or_fname) as algorithm_file:
		#             algorithm_config = yaml.load(algorithm_file, Loader=Python2to3Loader)
		#         assert issubclass(algorithm_config['class'], ServoingFittedQIterationAlgorithm)
		#         mean_returns = algorithm_config['mean_returns']
		#         thetas = algorithm_config['thetas']
		#     else:
		#         algorithm = algorithm_or_fname
		#         assert isinstance(algorithm, ServoingFittedQIterationAlgorithm)
		#         mean_returns = algorithm.mean_returns
		#         thetas = algorithm.thetas
		#     print("using parameters based on best returns")
		#     best_return, best_theta = max(zip(mean_returns, thetas))
		#     print(best_return)
		#     print(best_theta)
		#     self.theta = best_theta
		# self.target = np.asarray([331,257])
		self.target = np.asarray([320,240])
		with open('model_iter2.p', 'rb') as handle:
			model = pickle.load(handle)
			print("Reading old model parameters")

		with open('qfunction_iter3.p', 'rb') as handle:
			wl = pickle.load(handle)
			print("Reading old model parameters")
			print(wl)
			self._theta = wl.get("theta")
			self._w = self._theta[0]
			self._lambda_ = self._theta[1:3]
			self.lambda_ = self._theta[1:3]

		# Load model
		W0 = theano.shared(model.get("W0").astype(theano.config.floatX),"W0")
		W1 = theano.shared(model.get("W1").astype(theano.config.floatX),"W1")
		W2 = theano.shared(model.get("W2").astype(theano.config.floatX),"W2")
		b0 = theano.shared(model.get("b0").astype(theano.config.floatX),"b0")
		b1 = theano.shared(model.get("b1").astype(theano.config.floatX),"b1")
		b2 = theano.shared(model.get("b2").astype(theano.config.floatX),"b2")
		params = [W0,W1,W2,b0,b1,b2]
		y = T.dvector("y")
		u = T.dvector("u")

		# ypred = y + (T.dot(W1,y) + b1)*u[0] + (T.dot(W2,y) + b2)*u[1] + (T.dot(W0,y) + b0)
		ypred = y + (T.dot(W0,y) + b0)
		jacob = T.concatenate((T.reshape(T.dot(W1,y) + b1,[2,1]),T.reshape(T.dot(W2,y) + b2,[2,1])),axis = 1)

		self.predict = theano.function([y],[ypred])
		self.jacobian = theano.function([y],[jacob])

	@property
	def theta(self):
		# print("Theta")
		# print(self._theta)
		# print(self._w)
		# print(self._lambda_)
		assert all(self._theta == np.append(self._w, self._lambda_))
		if self.unweighted_features:
			assert all(self._w == self._w[0])
			theta = np.append(self.w[0], self.lambda_)
		else:
			theta = self._theta
		return theta

	@theta.setter
	def theta(self, theta):
		assert all(self._theta == np.append(self._w, self._lambda_))
		if self.unweighted_features:
			self.w = theta[0]
			self.lambda_ = theta[1:]
			assert all(self._w == self._w[0])
		else:
			self._theta[...] = theta

	@property
	def w(self):
		assert all(self._theta == np.append(self._w, self._lambda_))
		if self.unweighted_features:
			assert all(self._w == self._w[0])
		return self._w

	@w.setter
	def w(self, w):
		assert all(self._theta == np.append(self._w, self._lambda_))
		self._w[...] = w
		if self.unweighted_features:
			assert all(self._w == self._w[0])

	@property
	def lambda_(self):
		assert all(self._theta == np.append(self._w, self._lambda_))
		return self._lambda_

	@lambda_.setter
	def lambda_(self, lambda_):
		assert all(self._theta == np.append(self._w, self._lambda_))
		self._lambda_[...] = lambda_

	def phi(self, states, actions, preprocessed=False, with_constant=True):
		"""
		Corresponds to the linearized objective

		The following should be true
		phi = self.phi(states, actions)
		theta = np.append(self.w, self.lambda_)
		linearized_objectives = [self.linearized_objective(state, action, with_constant=False) for (state, action) in zip(states, actions)]
		objectives = phi.dot(theta)
		assert np.allclose(objectives, linearized_objectives)
		"""
		batch_size = len(states)
		# if preprocessed:
			# batch_image = np.array([obs[0] for (obs, target_obs) in states])
			# batch_target_image = np.array([target_obs[0] for (obs, target_obs) in states])
			# batch_target_image = np.array([self.target for (obs, target_obs) in states])
		batch_target_image = [self.target for i in range(len(states))]
		batch_u = np.array(actions)
		# else:
		# 	batch_image, = self.predictor.preprocess([obs[0] for (obs, target_obs) in states], batch_size=len(states))
		# 	batch_target_image, = self.predictor.preprocess([target_obs[0] for (obs, target_obs) in states],
		# 													batch_size=len(states))
		# 	batch_u = np.array([self.action_transformer.preprocess(action) for action in actions])
		# batch_target_feature = self.predictor.feature(batch_target_image, preprocessed=True)
		# batch_y_target = np.concatenate([f.reshape((f.shape[0], -1)) for f in batch_target_feature], axis=1)
		batch_y_target = np.array(batch_target_image)
		if self.alpha != 1.0:
			# batch_feature = self.predictor.feature(batch_image, preprocessed=True)
			# batch_y = np.concatenate([f.reshape((f.shape[0], -1)) for f in batch_feature], axis=1)
			batch_y = np.array(states)
			batch_y_target = self.alpha * batch_y_target + (1 - self.alpha) * batch_y
		# action_lin = np.zeros(self.action_space.shape)
		# u_lin = self.action_transformer.preprocess(action_lin)
		u_lin = [np.zeros([2]) for i in range(len(states))]

		# ------------------ Add Bilinear model here ------------------------- #
		# batch_jac, batch_next_feature = self.predictor.feature_jacobian(batch_image, np.array([u_lin] * batch_size), preprocessed=True)
		# batch_jac, batch_next_feature = [np.ones([2,2]) for i in range(len(states))] , [self.target for i in range(len(states))]
		batch_next_feature = [self.predict(i)[0] for i in states]
		batch_jac = [self.jacobian(i)[0] for i in states]
		# batch_J = np.concatenate(batch_jac, axis=1)
		# batch_y_next_pred = np.concatenate([f.reshape((f.shape[0], -1)) for f in batch_next_feature], axis=1)
		batch_J = np.array(batch_jac)
		batch_y_next_pred = np.array(batch_next_feature)
		# batch_z = batch_y_target - batch_y_next_pred + batch_J.dot(u_lin)
		batch_z = batch_y_target - batch_y_next_pred + np.array([i.dot(j) for i, j in zip(batch_J,u_lin)])

		# cs_repeats = np.cumsum(np.r_[0, self.repeats])
		# slices = [slice(start, stop) for (start, stop) in zip(cs_repeats[:-1], cs_repeats[1:])]
		# batch_A_split = np.array([np.einsum('nij,nik->njk', batch_J[:, s], batch_J[:, s]) for s in slices])
		# batch_b_split = np.array([np.einsum('nij,ni->nj', batch_J[:, s], batch_z[:, s]) for s in slices])
		# if with_constant:
		# 	batch_c_split = np.array([np.einsum('ni,ni->n', batch_z[:, s], batch_z[:, s]) for s in slices])
		# else:
		# 	batch_c_split = 0.0

		batch_A_split = np.einsum('nij,nik->njk', batch_J, batch_J)
		batch_b_split = np.einsum('nij,nj->ni', batch_J, batch_z)
		batch_c_split = np.einsum('ni,ni->n', batch_z, batch_z) 
		phi_errors = (np.einsum('nj,nj->n', np.einsum('njk,nk->nj', batch_A_split, batch_u), batch_u)
					  - 2 * np.einsum('nj,nj->n', batch_b_split, batch_u)
					  + batch_c_split).T
		phi_actions = batch_u ** 2
		phi = np.c_[phi_errors / self.repeats, phi_actions]
		return phi

	def pi(self, states, preprocessed=False):
		"""
		Corresponds to the linearized objective

		The following should be true
		actions_pi = self.pi(states)
		actions_act = [self.act(state) for state in states]
		assert np.allclose(actions_pi, actions_act)
		"""
		# if self.w.shape != (len(self.repeats),):
		# 	raise NotImplementedError
		# batch_size = len(states)
		# if preprocessed:
		# 	batch_image = np.array([obs[0] for (obs, target_obs) in states])
		# 	batch_target_image = np.array([target_obs[0] for (obs, target_obs) in states])
		# else:
		# 	batch_image, = self.predictor.preprocess([obs[0] for (obs, target_obs) in states], batch_size=len(states))
		# 	batch_target_image, = self.predictor.preprocess([target_obs[0] for (obs, target_obs) in states], batch_size=len(states))
		# batch_target_feature = self.predictor.feature(batch_target_image, preprocessed=True)
		# batch_y_target = np.concatenate([f.reshape((f.shape[0], -1)) for f in batch_target_feature], axis=1)
		# if self.alpha != 1.0:
		# 	batch_feature = self.predictor.feature(batch_image, preprocessed=True)
		# 	batch_y = np.concatenate([f.reshape((f.shape[0], -1)) for f in batch_feature], axis=1)
		# 	batch_y_target = self.alpha * batch_y_target + (1 - self.alpha) * batch_y
		# action_lin = np.zeros(self.action_space.shape)
		# u_lin = self.action_transformer.preprocess(action_lin)
		# batch_jac, batch_next_feature = self.predictor.feature_jacobian(batch_image, np.array([u_lin] * batch_size), preprocessed=True)
		# batch_J = np.concatenate(batch_jac, axis=1)
		# batch_y_next_pred = np.concatenate([f.reshape((f.shape[0], -1)) for f in batch_next_feature], axis=1)

		# batch_z = batch_y_target - batch_y_next_pred + batch_J.dot(u_lin)
		# cs_repeats = np.cumsum(np.r_[0, self.repeats])
		# slices = [slice(start, stop) for (start, stop) in zip(cs_repeats[:-1], cs_repeats[1:])]
		# batch_A_split = np.array([np.einsum('nij,nik->njk', batch_J[:, s], batch_J[:, s]) for s in slices])
		# batch_b_split = np.array([np.einsum('nij,ni->nj', batch_J[:, s], batch_z[:, s]) for s in slices])

		batch_target_image = [self.target for i in range(len(states))]

		batch_y_target = np.array(batch_target_image)

		if self.alpha != 1.0:
			batch_y = np.array(states)
			batch_y_target = self.alpha * batch_y_target + (1 - self.alpha) * batch_y
		u_lin = [np.zeros([2]) for i in range(len(states))]
		
		# batch_jac, batch_next_feature = [np.array([[1,2],[3,4]]) for i in range(len(states))] , [self.target for i in range(len(states))]
		batch_next_feature = [self.predict(i)[0] for i in states]
		batch_jac = [self.jacobian(i)[0] for i in states]

		batch_J = np.array(batch_jac)
		batch_y_next_pred = np.array(batch_next_feature)
		batch_z = batch_y_target - batch_y_next_pred + np.array([i.dot(j) for i, j in zip(batch_J,u_lin)])


		batch_A_split = np.einsum('nij,nik->njk', batch_J, batch_J)
		batch_b_split = np.einsum('nij,nj->ni', batch_J, batch_z)
		# batch_A = np.tensordot(batch_A_split, self.w / self.repeats, axes=(0, 0)) + np.diag(self.lambda_)
		# batch_b = np.tensordot(batch_b_split, self.w / self.repeats, axes=(0, 0))
		batch_A = self.w * batch_A_split + np.diag(self.lambda_)
		batch_b = self.w * batch_b_split
		batch_u = np.linalg.solve(batch_A, batch_b)

		# actions = np.array([self.action_transformer.deprocess(u) for u in batch_u])
		actions = batch_u
		# for action in actions:
		# 	self.action_space.clip(action, out=action)
		return actions

		
	def act(self, obs, action_lin=None):
		"""
		images and actions are in preprocessed units
		u is in processed units
		"""
		# assert isinstance(obs, dict)
		# image = obs[self.image_name]
		# target_image = obs[self.target_image_name]

		# features = self.predictor.feature([np.array([image, target_image])])
		# y, y_target = np.concatenate([f.reshape((f.shape[0], -1)) for f in features], axis=1)
		# y = np.asarray([obs.position.x,obs.position.y])
		y = obs
		y_target = self.target
		if self.alpha != 1.0:
			y_target = self.alpha * y_target + (1 - self.alpha) * y

		if action_lin is None:
			# action_lin = np.zeros(self.predictor.input_shapes[1])  # original units
			action_lin = np.zeros(2)  # original units

		# ------------ Add bilinear code here --------------------------- #
		# jac, next_feature = self.predictor.feature_jacobian([image, action_lin])  # Jacobian is in preprocessed units
		next_feature = self.predict(y)[0]
		jac = self.jacobian(y)[0]

		# jac, next_feature = np.ones([2,2]), y
		# J = np.concatenate(jac)
		J = jac
		# y_next_pred = np.concatenate([f.flatten() for f in next_feature])
		y_next_pred = next_feature

		# if self.w is None:
		#     WJ = J
		# elif self.w.shape == (len(self.repeats),):
		#     WJ = J * np.repeat(self.w / self.repeats, self.repeats)[:, None]
		# elif self.w.shape == (J.shape[0],):
		#     WJ = J * self.w[:, None]
		# elif self.w.shape == (J.shape[0], J.shape[0]):
		#     WJ = self.w.dot(J)
		# elif self.w.ndim == 2 and self.w.shape[0] == J.shape[0]:
		#     WJ = self.w.dot(self.w.T.dot(J))
		# else:
		#     raise ValueError('invalid weights w, %r' % self.w)
		WJ = J * np.repeat(self.w / self.repeats, self.repeats)[:, None]

		if self.use_constrained_opt:
			pass
			# import cvxpy
			# A = WJ.T.dot(J) + np.diag(self.lambda_)
			# b = WJ.T.dot(y_target - y_next_pred + J.dot(self.action_transformer.preprocess(action_lin)))

			# x = cvxpy.Variable(self.action_space.shape[0])
			# objective = cvxpy.Minimize((1. / 2.) * cvxpy.quad_form(x, A) - x.T * b)

			# action_low = self.action_transformer.preprocess(np.array(self.action_space.low))
			# action_high = self.action_transformer.preprocess(np.array(self.action_space.high))
			# if isinstance(self.action_space, (AxisAngleSpace, TranslationAxisAngleSpace)) and \
			#         self.action_space.axis is None:
			#     assert action_low[-1] ** 2 == action_high[-1] ** 2
			#     contraints = [cvxpy.sum_squares(x[-3:]) <= action_high ** 2]
			#     if isinstance(self.action_space, TranslationAxisAngleSpace):
			#         contraints.extend([action_low[:3] <= x[:3], x[:3] <= action_high[:3]])
			# else:
			#     constraints = [action_low <= x, x <= action_high]
			# prob = cvxpy.Problem(objective, constraints)

			# solved = False
			# for solver in [None, cvxpy.GUROBI, cvxpy.CVXOPT]:
			#     try:
			#         prob.solve(solver=solver)
			#     except cvxpy.error.SolverError:
			#         continue
			#     if x.value is None:
			#         continue
			#     solved = True
			#     break
			# if not solved:
			#     import IPython as ipy; ipy.embed()
			# u = np.squeeze(np.array(x.value), axis=1)
		else:
			try:
				u = np.linalg.solve(WJ.dot(J) + np.diag(self.lambda_),WJ.dot(y_target - y_next_pred + J.dot(action_lin)))  # preprocessed units
			except np.linalg.LinAlgError as e:
				print("Got linear algebra error. Returning zero action")
				# u = np.zeros(self.action_space.shape)
				u = np.zeros(2)

		# action = self.action_transformer.deprocess(u)
		action = u
		# if self.use_constrained_opt:
		#     try:
		#         assert self.action_space.contains((1 - 1e-6) * action)  # allow for rounding errors
		#     except AssertionError:
		#         import IPython as ipy; ipy.embed()
		# else:
		#     self.action_space.clip(action, out=action)
		return action
