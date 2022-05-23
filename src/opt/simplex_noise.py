import opensimplex
import time
import numpy as np

class SimplexNoise:
    def __init__(self, dim: int = 1, smoothness: int = 13, multiplier: float = 1.5):
        self.idx = 0
        self.dim = dim
        self.smoothness = smoothness
        self.multiplier = multiplier
        self.set_seed(int((time.time() % 1) * 10000000))

    def reset(self):
        self.idx = 0

    def set_seed(self, seed):
        opensimplex.seed(seed)

    def gen_noise_seq(self, n=100):
        return np.array([self.__call__() for _ in range(n)])

    def sample_parallel(self, n_samples, horizon, act_dim):
        self.idx += n_samples * horizon
        noise = opensimplex.noise2array(x=self.idx + np.arange(n_samples * horizon) / self.smoothness, y=np.array([0, 1]))
        reshaped_noise = np.reshape(noise, (n_samples, horizon, act_dim))
        clipped_noise = np.clip(reshaped_noise * self.multiplier, -1, 1).astype(dtype=np.float64)
        return clipped_noise

    def __call__(self):
        self.idx += 1
        noise = np.array([(opensimplex.noise2(x=self.idx / self.smoothness, y=i))
                         for i in range(self.dim)], dtype=np.float64)
        return np.clip(noise * self.multiplier, -1, 1).astype(dtype=np.float64)

    def __repr__(self) -> str:
        return f"SimplexNoise()"

if __name__ == "__main__":
    def gen_noise(n: int = 1000):
        noisegenerator = SimplexNoise(dim=1, smoothness=100, multiplier=2)
        return np.asarray([(noisegenerator() + 1) / 4 + 0.5 for _ in range(n)])

    def plotnoise():
        import matplotlib.pyplot as plt
        n = 1000
        plt.plot(np.arange(n), gen_noise(n=n)[:, ])
        plt.show()

    noisegenerator = SimplexNoise(dim=1, smoothness=100, multiplier=2)
    noisevec = noisegenerator.gen_noise_seq(10)
    plotnoise()
