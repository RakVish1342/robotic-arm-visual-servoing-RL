import numpy as np

states = [[331., 257.], [331., 257.], [331., 257.] ,[331., 257.]]
actions = [[ 0., 0.1137218 ], [ 0.05210123,  0.02254876] ,[-0.04190354,  0.11676107] ,[-0.73686205, -0.10354064]]

target = np.array([1,2])
alpha = 0.9
batch_size = len(states)

batch_target_image = [target for i in range(len(states))]
batch_u = np.array(actions)

batch_y_target = np.array(batch_target_image)
batch_y = np.array(states)
batch_y_target = alpha * batch_y_target + (1 - alpha) * batch_y
u_lin = [np.zeros([2]) for i in range(len(states))]
batch_jac, batch_next_feature = [np.array([[1,2],[3,4]]) for i in range(len(states))] , [target for i in range(len(states))]
batch_J = np.array(batch_jac)
batch_y_next_pred = np.array(batch_next_feature)
batch_z = batch_y_target - batch_y_next_pred + np.array([i.dot(j) for i, j in zip(batch_J,u_lin)])


batch_A_split = np.einsum('nij,nik->njk', batch_J, batch_J)
batch_b_split = np.einsum('nij,nj->ni', batch_J, batch_z)
batch_c_split = np.einsum('ni,ni->n', batch_z, batch_z) 

# print(batch_z)
# print(batch_J)
# print(batch_u)
# print(batch_A_split)
# print(batch_b_split)
print(batch_c_split)

# print(np.einsum('njk,nk->nj', batch_A_split, batch_u))
print(np.einsum('nj,nj->n', np.einsum('njk,nk->nj', batch_A_split, batch_u), batch_u))
print(np.einsum('nj,nj->n', batch_b_split, batch_u))
phi_errors = (np.einsum('nj,nj->n', np.einsum('njk,nk->nj', batch_A_split, batch_u), batch_u)
			  - 2 * np.einsum('nj,nj->n', batch_b_split, batch_u)
			  + batch_c_split).T
print(phi_errors)