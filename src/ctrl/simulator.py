import do_mpc

def make_simulator(model):
    # Obtain an instance of the do-mpc simulator class
    # and initiate it with the model:
    simulator = do_mpc.simulator.Simulator(model)

    # Set parameter(s):
    simulator.set_param(t_step=0.001)

    # Setup simulator:
    simulator.setup()

    return simulator