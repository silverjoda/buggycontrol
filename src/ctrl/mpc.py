import do_mpc
from casadi import sqrt
import numpy as np

def make_mpc_singletrack(model, waypoints=[3,3]):
    # Obtain an instance of the do-mpc MPC class
    # and initiate it with the model:
    mpc = do_mpc.controller.MPC(model)

    n_horizon = 7

    # Set parameters:
    setup_mpc = {
        'n_horizon': n_horizon,
        'n_robust': 1,
        't_step': 0.01,
        'nlpsol_opts': {'ipopt.print_level': 1, 'ipopt.sb': 'yes', 'print_time': 0}
    }
    mpc.set_param(**setup_mpc)

    lterm = sqrt((model.x['s_x'] - model.tvp['trajectory_set_point_x']) ** 2 + (
                model.x['s_y'] - model.tvp['trajectory_set_point_y']) ** 2)
    mterm = lterm
    mpc.set_objective(lterm=lterm, mterm=mterm)

    mpc.set_rterm(
        u_w=1e-3,
        u_d=1e-3
    )

    # Velocity bounds
    mpc.bounds['lower', '_x', 's_v'] = 0.1
    mpc.bounds['upper', '_x', 's_v'] = 3.0

    # Turn bounds
    mpc.bounds['lower', '_u', 'u_d'] = -0.45
    mpc.bounds['upper', '_u', 'u_d'] = 0.45

    # Wheel angular vel bound
    mpc.bounds['lower', '_u', 'u_w'] = 0.2
    mpc.bounds['upper', '_u', 'u_w'] = 300.0

    mpc.scaling['_u', 'u_w'] = 150

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(_):
        for k in range(n_horizon + 1):
            tvp_template['_tvp', k, 'trajectory_set_point_x'] = waypoints[0]
            tvp_template['_tvp', k, 'trajectory_set_point_y'] = waypoints[1]

        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()

    return mpc

def make_mpc_bicycle(model, waypoints=[3,3]):
    # Obtain an instance of the do-mpc MPC class
    # and initiate it with the model:
    mpc = do_mpc.controller.MPC(model)

    n_horizon = 8

    # Set parameters:
    setup_mpc = {
        'n_horizon': n_horizon,
        'n_robust': 1,
        't_step': 0.01,
        'nlpsol_opts' : {'ipopt.print_level': 1, 'ipopt.sb': 'yes', 'print_time': 0}
    }
    mpc.set_param(**setup_mpc)

    #x_tar, y_tar = 3., 3.

    lterm = sqrt((model.x['s_x'] - model.tvp['trajectory_set_point_x']) ** 2 + (model.x['s_y'] - model.tvp['trajectory_set_point_y']) ** 2)
    mterm = lterm
    mpc.set_objective(lterm=lterm, mterm=mterm)

    mpc.set_rterm(
        u_d=1e-3,
        u_D=1e-3
    )

    # Velocity bounds
    mpc.bounds['lower', '_x', 's_vx'] = 0.03
    mpc.bounds['upper', '_x', 's_vx'] = 2

    # Turn bounds
    mpc.bounds['lower', '_u', 'u_d'] = -0.45
    mpc.bounds['upper', '_u', 'u_d'] = 0.45

    # Whell angular vel bound from below
    mpc.bounds['lower', '_u', 'u_D'] = 0.0
    mpc.bounds['upper', '_u', 'u_D'] = 1.0

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(_):
        for k in range(n_horizon + 1):
            tvp_template['_tvp', k, 'trajectory_set_point_x'] = waypoints[0]
            tvp_template['_tvp', k, 'trajectory_set_point_y'] = waypoints[1]

        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()

    return mpc

def make_mpc_linmod_hybrid(model):
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

    lterm = sqrt((model.x['s_x'] - model.tvp['trajectory_set_point_x']) ** 2 + (model.x['s_y'] - model.tvp['trajectory_set_point_y']) ** 2)
    mterm = lterm
    mpc.set_objective(lterm=lterm, mterm=mterm)

    # mpc.set_rterm(
    #     u_d=1e-3,
    #     u_D=1e-3
    # )

    mpc.bounds['lower', '_x', 'x'] = -np.ones((15, 1))
    mpc.bounds['upper', '_x', 'x'] = np.ones((15, 1))

    mpc.bounds['lower', '_u', 'u'] = -np.ones((6, 1))
    mpc.bounds['upper', '_u', 'u'] = np.ones((6, 1))

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(_):
        for k in range(n_horizon + 1):
            tvp_template['_tvp', k, 'trajectory_set_point_x'] = 0
            tvp_template['_tvp', k, 'trajectory_set_point_y'] = 0

        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()

    return mpc