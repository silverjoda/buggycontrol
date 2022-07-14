import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data_path = "/home/silverjoda/SW/tmp-data"
lsa_data_path = os.path.join(data_path, "lsa_data.npy")
e2e_data_path = os.path.join(data_path, "e2e_data.npy")

lsa_data = np.load(lsa_data_path, allow_pickle=True)
e2e_data = np.load(e2e_data_path, allow_pickle=True)

lsa_param_list = []
lsa_rew_list = []
for i in range(len(lsa_data)):
    lsa_param_list.append(list(lsa_data[i, 1]))
    lsa_rew_list.append([lsa_data[i, 0]])
lsa_param_vec = np.array(lsa_param_list)
lsa_rew_vec = np.array(lsa_rew_list)
lsa_param_rew_vec = np.concatenate((lsa_param_vec, lsa_rew_vec), axis=1)

e2e_param_list = []
e2e_rew_list = []
for i in range(len(e2e_data)):
    e2e_param_list.append(list(e2e_data[i, 1]))
    e2e_rew_list.append([e2e_data[i, 0]])
e2e_param_vec = np.array(e2e_param_list)
e2e_rew_vec = np.array(e2e_rew_list)
e2e_param_rew_vec = np.concatenate((e2e_param_vec, e2e_rew_vec), axis=1)

e2e_param_df = pd.DataFrame(e2e_param_rew_vec)
e2e_rew_df = pd.DataFrame(e2e_rew_vec)

x = e2e_param_df.loc[:, :].values
x = StandardScaler().fit_transform(x)

rew_normed = StandardScaler().fit_transform(e2e_rew_df.loc[:, :].values)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, e2e_rew_df], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

rew_colors = (e2e_rew_df.loc[:, 0] + np.min(e2e_rew_df.loc[:, 0])) * 10
ax.scatter(principalDf.loc[:, 'principal component 1']
           , principalDf.loc[:, 'principal component 2']
           , c = rew_colors
           , s = 50
           , cmap='hot')
ax.grid()
plt.show()

#lsa_data_shape = lsa_param_vec.shape
e2e_data_shape = e2e_param_vec.shape

#print(f"Lsa data shape: {lsa_data_shape}, e2e data shape: {e2e_data_shape}")

# Do PCA


#plt.scatter(lsa_data[:, 0], lsa_data[:, 1])
#plt.show()