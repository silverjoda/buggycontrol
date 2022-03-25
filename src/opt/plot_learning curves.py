import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

def plot_basic():
    file_path_def = "/home/silverjoda/SW/buggy_ws/src/buggycontrol/src/opt/logs/def"
    file_path_sym = "/home/silverjoda/SW/buggy_ws/src/buggycontrol/src/opt/logs/be2sym"

    def_list = []
    for i in range(5):
        fp = os.path.join(file_path_def, "{}".format(i), "evaluations.npz")
        data = np.load(fp)
        def_list.append(np.mean(data.f.results, axis=1)[:49])
    def_arr = np.mean(np.array(def_list), axis=0)

    sym_list = []
    for i in range(5):
        fp = os.path.join(file_path_sym, "{}".format(i), "evaluations.npz")
        data = np.load(fp)
        df = np.zeros(50)
        df[np.arange(0,50,2)] = np.mean(data.f.results, axis=1)
        df[np.arange(1,50,2)] = np.mean(data.f.results, axis=1)
        sym_list.append(df[:49])
    sym_arr = np.mean(np.array(sym_list), axis=0)

    N = len(def_arr)
    plt.plot(range(N), def_arr)
    plt.plot(range(N), sym_arr)
    plt.show()

if __name__ == "__main__":
    plot_basic()