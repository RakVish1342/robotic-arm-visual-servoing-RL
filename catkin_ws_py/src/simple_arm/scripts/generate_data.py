# import yaml
import argparse
import os
import time
import numpy as np
# from base import ServoingOptimizationAlgorithm
from fqi import ServoingFittedQIterationAlgorithm
from arm_env import RoboticArm
from servoing_policy import ServoingPolicy 
from math import pi
import pickle
from random import choice

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--output_dir', '-o', type=str, default=None)
	parser.add_argument('--sampling_iters', '-i', type=int, default=20)

	args = parser.parse_args()
	env = RoboticArm(); 
	
	states, observations, prev_observations, actions, rewards = [], [], [], [], []

	num_steps = 1000

	iter_ = 0 
	reward = 0
	errors_row_format = '{:>30}{:>15.4f}'
	state = None
	while iter_ <= args.sampling_iters:
		# state = None
		state = np.concatenate((choice([np.random.uniform(0,0.6,1),np.random.uniform(5.6,6.284,1)]),np.random.uniform(1.1,2.1,1)))
		obs = env.reset(state)
		time.sleep(0.5)
		for step_iter in range(num_steps):
			try:
				# state = env.get_state()
				action = np.random.uniform(-pi/12,pi/12,size = [2,])
				prev_obs = obs
				s, obs, reward, episode_done = env.step(action)  # action is updated in-place if needed
				# time.sleep(0.1)
				print(errors_row_format.format(str((iter_, step_iter)), reward))
				if reward > -0.9:
					states.append(s)
					observations.append(obs)
					prev_observations.append(prev_obs)
					actions.append(action)
					rewards.append(reward)
				else:
					# state = None
					state = np.concatenate((choice([np.random.uniform(0,0.6,1),np.random.uniform(5.6,6.284,1)]),np.random.uniform(1.1,2.1,1)))
					obs = env.reset(state)
					time.sleep(0.5)
			except KeyboardInterrupt:
				break	
		iter_ += 1


	itemlist = {"states" : states, "observations" : observations, "prev_observations" : prev_observations, "actions" : actions, "rewards" : rewards}
	print("Saving data ... in " + os.getcwd())
	with open('data_file.p', 'wb') as fp:
		pickle.dump(itemlist, fp)

if __name__ == '__main__':
	main()
