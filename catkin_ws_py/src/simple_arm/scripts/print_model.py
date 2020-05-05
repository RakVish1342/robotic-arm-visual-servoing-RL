import pickle

def main():

	with open('qfunction_iter3.p', 'rb') as handle:
		model = pickle.load(handle)
		print("Reading q function parameters")
		print(model)

	with open('model_iter2.p', 'rb') as handle:
		model = pickle.load(handle)
		print("Reading bilinear parameters")
		print(model)

if __name__ == '__main__':
	main()


# \textbf{Bilinear Model} \\
# \\
# W_0 = 
# \begin{bmatrix} 
# 1.58003657e^{-02} & 5.53979803e^{-04} \\
# 1.05947522e^{-05} & 1.59736264e^{-03} \\
# \end{bmatrix}
# \\
# W_1 = 
# \begin{bmatrix} 
# 0.00497141 & -0.02230702\\
# 0.00497141 & -0.02230702\\
# \end{bmatrix}
# \\
# W_2 = 
# \begin{bmatrix} 
# 0.00497141 & -0.02230702 \\ 
# -0.01345574 & -0.06434786 \\
# \end{bmatrix}
# \\
# b_0 = \begin{bmatrix} -5.16923313 \\ -5.1888254 \\ \end{bmatrix}
# \\
# b_1 = \begin{bmatrix} 249.46013493 \\ -18.63554689 \\ \end{bmatrix}
# \\
# b_2 = \begin{bmatrix} -18.53414016 \\ -212.69774129 \\ \end{bmatrix}
# \\
# \\
# \textbf{Weights from FQI}
# \\
# \theta = \begin{bmatrix} w \\ \lambda_1 \\ \lambda_2 \end{bmatrix}
# = \begin{bmatrix} -9.21447146e^{-24} \\   -2.07722530e^{-18} \\  -1.04664147e^{-18} \end{bmatrix}