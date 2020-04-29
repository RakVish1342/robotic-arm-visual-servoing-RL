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
import numpy as np

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
	with open('qfunction_iter3.p', 'rb') as handle:
		model = pickle.load(handle)
		print("Reading old model parameters")
		print(model)

	args = parser.parse_args()
	env = RoboticArm(); 
	pol = ServoingPolicy()    

	if model is not None:
		pol.theta = model.get("theta")


	try:
		while True:
			y = env.box_loc
			print("Current box_loc:\t"+str(y[0])+'\t'+str(y[1]))
			
			if y[0] < 640 and y[1] < 480:
				A_p = pol.pi([y], preprocessed=True)
				A_p = np.clip(A_p,a_min = -0.4,a_max = 0.4)
			else: 
				A_p = np.random.randint(0,200,[2,])*0.01-1
				# time.sleep(0.3)
			
			print(A_p)
			# print("Action:\t"+str(A_p[0])+'\t'+str(A_p[1]))
			_,_,_,_ = env.step(A_p[0])
			time.sleep(0.1)
	except KeyboardInterrupt:
		print('interrupted!')

if __name__ == '__main__':
	main()