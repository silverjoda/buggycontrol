import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_vec_and_rew(data_path):
    data = np.load(data_path, allow_pickle=True)

    param_list = []
    rew_list = []
    for i in range(len(data)):
        param_list.append(list(data[i, 1]))
        rew_list.append([data[i, 0]])
    param_vec = np.array(param_list)
    rew_vec = np.array(rew_list)
    param_rew_vec = np.concatenate((param_vec, rew_vec), axis=1)

    return param_vec, rew_vec, param_rew_vec

def process_and_plot(param, rew):
    param_df = pd.DataFrame(param)
    rew_df = pd.DataFrame(rew)

    x = param_df.loc[:, :].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['pc_1', 'pc_2'])

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Component 1', fontsize=15)
    ax.set_ylabel('Component 2', fontsize=15)
    #ax.set_title('2 component PCA', fontsize=20)

    rew_normed = StandardScaler().fit_transform(rew_df.loc[:, :].values)
    rew_colors = (rew_normed + np.min(rew_normed)) * 10
    ax.scatter(principalDf.loc[:, 'pc_1']
               , principalDf.loc[:, 'pc_2']
               , c=rew_colors
               , s=50
               , cmap='hot')
    ax.grid()

    return

def process_and_plot_experimental(param, rew):
    param_df = pd.DataFrame(param)
    rew_df = pd.DataFrame(rew)

    # Do initial pca
    x = param_df.loc[:, :].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['pc_1', 'pc_2'])

    # Do repeated pca
    principalDf = pd.concat([principalDf, rew_df])
    x_2 = principalDf.loc[:, :].values
    x_2 = StandardScaler().fit_transform(x_2)
    pca_2 = PCA(n_components=2)
    principalComponents_2 = pca_2.fit_transform(x_2)
    principalDf_2 = pd.DataFrame(data=principalComponents_2, columns=['pc_1', 'pc_2'])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('pc_1', fontsize=15)
    ax.set_ylabel('pc_2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    rew_normed = StandardScaler().fit_transform(rew_df.loc[:, :].values)
    rew_colors = (rew_normed + np.min(rew_normed)) * 10
    ax.scatter(principalDf_2.loc[:, 'pc_1']
               , principalDf_2.loc[:, 'pc_2']
               , c=rew_colors
               , s=50
               , cmap='hot')
    ax.grid()
    return

# Load data
data_path = f"/home/{os.environ.get('USERNAME')}/SW/data"

lsa_data_path = os.path.join(data_path, "lsa_data.npy")
e2e_data_path = os.path.join(data_path, "e2e_data.npy")

lsa_param_vec, lsa_rew_vec, lsa_param_rew_vec = get_vec_and_rew(lsa_data_path)
e2e_param_vec, e2e_rew_vec, e2e_param_rew_vec = get_vec_and_rew(e2e_data_path)

print(f"Lsa data shape: {lsa_param_vec.shape}, e2e data shape: {e2e_param_vec.shape}")

# Do PCA
process_and_plot(e2e_param_vec, e2e_rew_vec)
process_and_plot(lsa_param_vec, lsa_rew_vec)
#process_and_plot_experimental(e2e_param_vec, e2e_rew_vec)
plt.show()