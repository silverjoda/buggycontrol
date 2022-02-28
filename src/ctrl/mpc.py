import do_mpc

def make_mpc(model):
    # Obtain an instance of the do-mpc MPC class
    # and initiate it with the model:
    mpc = do_mpc.controller.MPC(model)

    # Set parameters:
    setup_mpc = {
        'n_horizon': 20,
        'n_robust': 1,
        't_step': 0.005,
    }
    mpc.set_param(**setup_mpc)

    # Configure objective function:
    mterm = (_x['C_b'] - 0.6)**2    # Setpoint tracking
    lterm = (_x['C_b'] - 0.6)**2    # Setpoint tracking

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(F=0.1, Q_dot = 1e-3) # Scaling for quad. cost.

    # State and input bounds:
    mpc.bounds['lower', '_x', 'C_b'] = 0.1
    mpc.bounds['upper', '_x', 'C_b'] = 2.0
    ...

    mpc.setup()

    return mpc