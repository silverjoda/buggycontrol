import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

def plot_basic():
    file_path_def = "/home/tim/SW/buggy_ws/src/buggycontrol/src/opt/logs/def"
    file_path_sym = "/home/tim/SW/buggy_ws/src/buggycontrol/src/opt/logs/sym"

    def_list = []
    for i in range(10):
        fp = os.path.join(file_path_def, "{}".format(i), "evaluations.npz")
        data = np.load(fp)
        def_list.append(data.f.results.mean(axis=1))
    def_arr = np.array(def_list).mean(axis=0)

    sym_list = []
    for i in range(10):
        fp = os.path.join(file_path_sym, "{}".format(i), "evaluations.npz")
        data = np.load(fp)
        sym_list.append(data.f.results.mean(axis=1))
    sym_arr = np.array(sym_list).mean(axis=0)

    N = len(def_arr)
    plt.plot(range(N), def_arr)
    plt.plot(range(N), sym_arr)
    plt.show()

if __name__ == "__main__":
    plot_basic()