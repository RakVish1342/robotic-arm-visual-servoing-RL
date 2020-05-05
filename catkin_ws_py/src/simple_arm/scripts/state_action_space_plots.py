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
import matplotlib.pyplot as plt
import seaborn as sns

def main():

	with open('qfunction_iter3.p', 'rb') as handle:
		model = pickle.load(handle)
		print("Reading old model parameters")
		print(model)

	pol = ServoingPolicy()    

	if model is not None:
		pol.theta = model.get("theta")
		pass

	# Actions
	x1 = np.zeros([640,480])
	x2 = np.zeros([640,480])

	for i in range(480):
		for j in range(640):
			y = np.array([i,j])
			# print("Current box_loc:\t"+str(y[0])+'\t'+str(y[1]))
			
			if y[0] < 480 and y[1] < 640:
				A_p = pol.pi([y], preprocessed=True)
				# A_p = np.clip(A_p,a_min = -0.4,a_max = 0.4)
			else: 
				A_p = np.random.randint(0,200,[2,])*0.01-1
				# time.sleep(0.3)			

			x1[j,i] = A_p[0][0]
			x2[j,i] = A_p[0][1]

	f = plt.figure(1)
	sns.heatmap(x1, cmap="RdYlGn", yticklabels=False,xticklabels=False)
	f.show()
	
	g = plt.figure(2)
	sns.heatmap(x2, cmap="RdYlGn", yticklabels=False,xticklabels=False)
	g.show()

	raw_input()

if __name__ == '__main__':
	main()

	 # Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r