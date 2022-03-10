import do_mpc

def make_simulator(model):
    # Obtain an instance of the do-mpc simulator class
    # and initiate it with the model:
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        "integration_tool" : "cvodes",
        "abstol" : 1e-10,
        "reltol" : 1e-10,
        "t_step" : 0.001
    }

    # Set parameter(s):
    simulator.set_param(**params_simulator)

    tvp_template = simulator.get_tvp_template()

    def tvp_fun(t_now):
        return tvp_template

    simulator.set_tvp_fun(tvp_fun)

    # Setup simulator:
    simulator.setup()

    return simulator