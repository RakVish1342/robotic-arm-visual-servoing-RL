import pickle
import os
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
# theano.config.optimizer='fast_compile'
# theano.config.exception_verbosity='high'
# theano.config.compute_test_value = 'warn'

def main():  

	print(os.getcwd())
	with open('outfile_reset_upd.p', 'rb') as handle:
		dump = pickle.load(handle)

	try:
		with open('model_iter2.p', 'rb') as handle:
			model = pickle.load(handle)
			print("Reading old model parameters")
			print(model)
			# print(model)
	except IOError:
		model = None
		print("No previous model found. Training from scratch")


	observations = dump.get("observations")
	prev_observations = dump.get("prev_observations")
	actions = dump.get("actions")

	stop_iter = len(observations)
	print("Number of observations:" + str(stop_iter))

	rng = np.random
	# Load model
	W0 = theano.shared(rng.uniform(0,0.001,[2,2]).astype(theano.config.floatX),"W0")
	W1 = theano.shared(rng.uniform(0,0.001,[2,2]).astype(theano.config.floatX),"W1")
	W2 = theano.shared(rng.uniform(0,0.001,[2,2]).astype(theano.config.floatX),"W2")
	b0 = theano.shared(np.zeros(2).astype(theano.config.floatX),"b0")
	b1 = theano.shared(np.zeros(2).astype(theano.config.floatX),"b1")
	b2 = theano.shared(np.zeros(2).astype(theano.config.floatX),"b2")

	if model is not None:
		print("Setting old model parameters")
		W0.set_value(model.get("W0").astype(theano.config.floatX))
		W1.set_value(model.get("W1").astype(theano.config.floatX))
		W2.set_value(model.get("W2").astype(theano.config.floatX))
		b0.set_value(model.get("b0").astype(theano.config.floatX))
		b1.set_value(model.get("b1").astype(theano.config.floatX))
		b2.set_value(model.get("b2").astype(theano.config.floatX))


	params = [W0,W1,W2,b0,b1,b2]

	# Variables
	y = T.dvector("y")
	y.tag.test_value = rng.rand(2,)
	u = T.dvector("u")
	u.tag.test_value = rng.rand(2,)
	yactual = T.dvector("yactual")
	yactual.tag.test_value = rng.rand(2,)
	# Loss equation
	ypred = y + (T.dot(W1,y) + b1)*u[0] + (T.dot(W2,y) + b2)*u[1] + (T.dot(W0,y) + b0)
	loss = ((ypred - yactual)**2).sum()

	# Gradients
	# gW0,gW1,gW2,gb0,gb1,gb2 = T.grad(loss, [W0,W1,W2,b0,b1,b2])
	all_grads = T.grad(loss, [W0,W1,W2,b0,b1,b2])

	# Adam Updates
	# learning_rate=0.001
	beta1=0.9
	beta2=0.999
	epsilon=1e-8

	t_prev = theano.shared(0.)
	updates = OrderedDict()
	one = T.constant(1) # Using theano constant to prevent upcasting of float32
	learning_rate = theano.tensor.scalar(name='learning_rate')
	learning_rate.tag.test_value = 0.1

	# ADAM updates
	t = t_prev + 1
	a_t = learning_rate*T.sqrt(one-beta2**t)/(one-beta1**t)
	# i = 1
	for param, g_t in zip(params, all_grads):
		# print(i)
		value = param.get_value(borrow=True)
		# print(value)
		m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
							   broadcastable=param.broadcastable)
		v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
							   broadcastable=param.broadcastable)
		m_t = beta1*m_prev + (one-beta1)*g_t
		v_t = beta2*v_prev + (one-beta2)*g_t**2
		step = a_t*m_t/(T.sqrt(v_t) + epsilon)
		updates[m_prev] = m_t
		updates[v_prev] = v_t
		updates[param] = param - step
		# i += 1

	updates[t_prev] = t

	train = theano.function([y,u,yactual,learning_rate],[loss], updates = updates)
	# Train function
	predict = theano.function([y,u],[ypred])

	# Train model
	stepsize = 1000
	gamma = 0.9
	base_lr = 0.001
	print("Starting training...")

	min_value = 10000000000
	for i in range(100):
		train_data_gen = iter([[x,a,xd] for x,a,xd in zip(prev_observations,actions,observations)])
		iter_ = 0 
		print(i)
		while iter_ < stop_iter:
			current_step = iter_ // stepsize
			lr = base_lr * gamma ** current_step
			train_batch_data = tuple(next(train_data_gen))
			loss_value = train(*(train_batch_data + (lr,)))
			if abs(loss_value[0]) < min_value:
				print("Min Loss:" + str(loss_value[0]))
				min_value = abs(loss_value[0])
			iter_ += 1
	# Validate model

	model_params = {"W0" : W0.get_value(),
					"W1" : W1.get_value(),
					"W2" : W2.get_value(),
					"b0" : b0.get_value(),
					"b1" : b1.get_value(),
					"b2" : b2.get_value()}
	# Save model
	print("Saving model parameters ... in " + os.getcwd())
	print(model_params)
	with open('model_iter3.p', 'wb') as fp:
		pickle.dump(model_params, fp)

if __name__ == '__main__':
	main()