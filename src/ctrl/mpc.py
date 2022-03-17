import do_mpc
from casadi import sqrt

def make_mpc_singletrack(model):
    # Obtain an instance of the do-mpc MPC class
    # and initiate it with the model:
    mpc = do_mpc.controller.MPC(model)

    n_horizon = 3

    # Set parameters:
    setup_mpc = {
        'n_horizon': n_horizon,
        'n_robust': 1,
        't_step': 0.01,
        'nlpsol_opts': {'ipopt.print_level': 1, 'ipopt.sb': 'yes', 'print_time': 0}
    }
    mpc.set_param(**setup_mpc)

    x_tar, y_tar = 3, 1

    lterm = sqrt((model.x['s_x'] - x_tar) ** 2 + (model.x['s_y'] - y_tar) ** 2)
    mterm = lterm
    mpc.set_objective(lterm=lterm, mterm=mterm)

    mpc.set_rterm(
        u_w=1e-3,
        u_d=1e-3
    )

    # Velocity bounds
    mpc.bounds['lower', '_x', 's_v'] = 0.03

    # Turn bounds
    mpc.bounds['lower', '_u', 'u_d'] = -0.38
    mpc.bounds['upper', '_u', 'u_d'] = 0.38

    # Wheel angular vel bound from below
    mpc.bounds['lower', '_u', 'u_w'] = 0.01
    mpc.bounds['upper', '_u', 'u_w'] = 30.0

    # tvp_template = mpc.get_tvp_template()
    #
    # def tvp_fun(_):
    #     for k in range(n_horizon + 1):
    #         tvp_template['_tvp', k, 'trajectory_set_point_x'] = 10
    #         tvp_template['_tvp', k, 'trajectory_set_point_y'] = 10
    #
    #     return tvp_template
    #
    # mpc.set_tvp_fun(tvp_fun)
    mpc.setup()

    return mpc

def make_mpc_bicycle(model):
    # Obtain an instance of the do-mpc MPC class
    # and initiate it with the model:
    mpc = do_mpc.controller.MPC(model)

    n_horizon = 3

    # Set parameters:
    setup_mpc = {
        'n_horizon': n_horizon,
        'n_robust': 1,
        't_step': 0.01,
        'nlpsol_opts' : {'ipopt.print_level': 1, 'ipopt.sb': 'yes', 'print_time': 0}
    }
    mpc.set_param(**setup_mpc)

    x_tar, y_tar = 3., 3.

    lterm = sqrt((model.x['s_x'] - x_tar) ** 2 + (model.x['s_y'] - y_tar) ** 2)
    mterm = lterm
    mpc.set_objective(lterm=lterm, mterm=mterm)

    mpc.set_rterm(
        u_d=1e-3,
        u_D=1e-3
    )

    # Velocity bounds
    mpc.bounds['lower', '_x', 's_vx'] = 0.03

    # Turn bounds
    mpc.bounds['lower', '_u', 'u_d'] = -0.38
    mpc.bounds['upper', '_u', 'u_d'] = 0.38

    # Whell angular vel bound from below
    mpc.bounds['lower', '_u', 'u_D'] = 0.0
    mpc.bounds['upper', '_u', 'u_D'] = 1.0

    # tvp_template = mpc.get_tvp_template()
    #
    # def tvp_fun(_):
    #     for k in range(n_horizon + 1):
    #         tvp_template['_tvp', k, 'trajectory_set_point_x'] = 10
    #         tvp_template['_tvp', k, 'trajectory_set_point_y'] = 10
    #
    #     return tvp_template
    #
    # mpc.set_tvp_fun(tvp_fun)
    mpc.setup()

    return mpc