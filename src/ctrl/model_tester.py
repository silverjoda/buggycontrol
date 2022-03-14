from casadi import *

from src.ctrl.model import make_bicycle_model, make_singletrack_model
from src.ctrl.simulator import make_simulator

#model = make_bicycle_model()
model = make_singletrack_model()
simulator = make_simulator(model)

x0 = np.array([0.001, 0.001, 0.0, 0.00, 0.00, 0.00]).reshape(-1,1)
simulator.x0 = x0

for i in range(100):
    u0 = np.array([np.random.randn() * 0.4, np.random.rand() * 10]).reshape(-1,1)
    simulator.make_step(u0)
    print(i)
