import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import os


def plot_basic():
    file_path_def = f"{os.path.expanduser('~')}/SW/buggy_ws/src/buggycontrol/src/opt/logs/be1"
    file_path_sym = f"{os.path.expanduser('~')}/SW/buggy_ws/src/buggycontrol/src/opt/logs/be2"

    def_list = []
    for i in range(4):
        fp = os.path.join(file_path_def, "{}".format(i), "evaluations.npz")
        data = np.load(fp)
        data_results = data.f.results
        def_list.append(np.mean(data_results, axis=1)[:80])
    def_arr = np.mean(np.array(def_list), axis=0)

    sym_list = []
    for i in range(4):
        fp = os.path.join(file_path_sym, "{}".format(i), "evaluations.npz")
        data = np.load(fp)
        data_results = data.f.results
        sym_list.append(np.mean(data_results, axis=1)[:80])
    sym_arr = np.mean(np.array(sym_list), axis=0)

    N = len(def_arr)
    plt.plot(range(N), def_arr)
    plt.plot(range(N), sym_arr)
    plt.show()

if __name__ == "__main__":
    plot_basic()