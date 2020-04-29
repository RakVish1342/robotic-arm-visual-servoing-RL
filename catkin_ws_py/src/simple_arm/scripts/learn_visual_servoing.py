# import yaml
import argparse
import os
import time
# from base import ServoingOptimizationAlgorithm
from fqi import ServoingFittedQIterationAlgorithm
from arm_env import RoboticArm
from servoing_policy import ServoingPolicy 
from math import pi
import pickle

def main():
	parser = argparse.ArgumentParser()
	# parser.add_argument('predictor_fname', type=str)
	# parser.add_argument('algorithm_fname', type=str)
	# parser.add_argument('--algorithm_init_fname', type=str, default=None)
	# parser.add_argument('--output_dir', '-o', type=str, default=None)
	# parser.add_argument('--visualize', '-v', type=int, default=None)
	# parser.add_argument('--record_file', '-r', type=str, default=None)
	# parser.add_argument('--cv2_record_file', type=str, default=None)
	# parser.add_argument('--w_init', type=float, nargs='+', default=1.0)
	# parser.add_argument('--lambda_init', type=float, nargs='+', default=1.0)
	try:
		with open('qfunction_iter2.p', 'rb') as handle:
			model = pickle.load(handle)
			print("Reading old model parameters")
			print(model)
			# print(model)
	except IOError:
		model = None
		print("No previous model found. Training from scratch")

	args = parser.parse_args()
	env = RoboticArm(); 
	pol = ServoingPolicy()

	if model is not None:
		pol.theta = model.get("theta")

	alg = ServoingFittedQIterationAlgorithm(env,pol,10,5,num_trajs=20,num_steps = 100,skip_validation = True)
	# alg = ServoingFittedQIterationAlgorithm(env,pol,5,2,num_trajs=3,num_steps = 5,skip_validation = True)
	alg.run();

	print("Saving Theta")
	params = {"theta" : pol.theta}
	print(params)
	with open('qfunction_iter3.p', 'wb') as fp:
		pickle.dump(params, fp)

	print('Hello World')
	# time.sleep(5)
	# print(env.get_state())
	# env.set_state([0.5,pi/2 + 0.5])
	# time.sleep(5)
	# print(env.get_state())
	# print("Resetting")
	# env.reset()
	# # time.sleep(1)
	# print(env.get_state())
	# print("Closing")
	# env.close()
	# print("Its Closed")

if __name__ == '__main__':
	main()


	# [1.06902958e-06 2.64863279e-01 1.39750894e-01]
	# 9.58883898381

	# [1.06386789e-06 2.53093307e-01 1.66443545e-01]
	# 9.58991837766
