import do_mpc
from casadi import sqrt

def make_mpc(model):
    # Obtain an instance of the do-mpc MPC class
    # and initiate it with the model:
    mpc = do_mpc.controller.MPC(model)

    # Set parameters:
    setup_mpc = {
        'n_horizon': 200,
        'n_robust': 1,
        't_step': 0.01,
    }
    mpc.set_param(**setup_mpc)

    lterm = sqrt((model.x['s_x'] - model.tvp['trajectory_set_point_x']) ** 2 + (model.x['s_y'] - model.tvp['trajectory_set_point_y']) ** 2)
    mterm = lterm
    mpc.set_objective(lterm=lterm, mterm=mterm)

    mpc.set_rterm(
        u_d=1e-2,
        u_w=1e-2
    )

    # Turn bounds
    mpc.bounds['lower', '_u', 'u_d'] = -0.38
    mpc.bounds['upper', '_u', 'u_d'] = 0.38

    # Whell angular vel bound from below
    mpc.bounds['lower', '_u', 'u_w'] = 0.0

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        for k in range(200 + 1):
            tvp_template['_tvp', k, 'trajectory_set_point_x'] = 10
            tvp_template['_tvp', k, 'trajectory_set_point_y'] = 10

        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()

    return mpc