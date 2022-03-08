import do_mpc

def make_mpc(model):
    # Obtain an instance of the do-mpc MPC class
    # and initiate it with the model:
    mpc = do_mpc.controller.MPC(model)

    # Set parameters:
    setup_mpc = {
        'n_horizon': 200,
        'n_robust': 1,
        't_step': 0.005,
    }
    mpc.set_param(**setup_mpc)

    # Configure objective function:
    mterm = (model.x['s_b'] - 0.6)**2  # Setpoint tracking
    lterm = (model.x['s_b'] - 0.6)**2  # Setpoint tracking

    mpc.set_objective(mterm=mterm, lterm=lterm)
    #mpc.set_rterm(F=0.1, Q_dot = 1e-3) # Scaling for quad. cost.

    # Turn bounds
    mpc.bounds['lower', '_u', 'u_d'] = -0.38
    mpc.bounds['upper', '_u', 'u_d'] = 0.38

    # Whell angular vel bound from below
    mpc.bounds['low', '_u', 'u_w'] = 0.0

    mpc.setup()

    return mpc