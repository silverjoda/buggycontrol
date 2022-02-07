from opensimplex import OpenSimplex
import time
import numpy as np

class SimplexNoise:
    def __init__(self, dim: int = 1, smoothness: int = 13, multiplier: float = 1.5):
        self.idx = 0
        self.dim = dim
        self.smoothness = smoothness
        self.multiplier = multiplier
        self.noisefun = OpenSimplex(seed=int((time.time() % 1) * 10000000))

    def reset(self):
        self.idx = 0

    def set_seed(self, seed):
        self.noisefun = OpenSimplex(seed=seed)

    def gen_noise_seq(self, n=100):
        return np.array([self.__call__() for _ in range(n)])

    def __call__(self):
        self.idx += 1
        noise = np.array([(self.noisefun.noise2d(x=self.idx / self.smoothness, y=i))
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
    #plotnoise()
