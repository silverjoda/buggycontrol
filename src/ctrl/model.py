import do_mpc
import numpy as np
from numpy import cos, sin, arctan, abs, tan, sqrt, arccos

def make_model():
    # Obtain an instance of the do-mpc model class
    # and select time discretization:
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # Physical model:
    m = 1200 # Vehicle mass (kg)
    I = 2688 # Yaw moment of inertia (kgm-2)
    l_f = 1.4 # Front axle-CG distance (m)
    l_r = 1.6 # Rear axle-CG distance (m)
    p = 0.33 # Radius of wheels (m)
    g = 9.81 # Gravitational constant (ms-2)

    # Shaping coefficients for lateral dynamics
    cDy = 1
    cBy = 6.9
    cCy = 1.8
    cEy = 0.1

    cDx = 1
    cBx = 15
    cCx = 1.7
    cEx = -0.5

    # Introduce new states
    s_b = model.set_variable(var_type='_x', var_name='s_b', shape=(1,1))
    s_v = model.set_variable(var_type='_x', var_name='s_v', shape=(1,1))
    s_r = model.set_variable(var_type='_x', var_name='s_r', shape=(1,1))
    s_x = model.set_variable(var_type='_x', var_name='s_x', shape=(1,1))
    s_y = model.set_variable(var_type='_x', var_name='s_y', shape=(1,1))
    s_phi = model.set_variable(var_type='_x', var_name='s_phi', shape=(1,1))

    # Set control inputs to target turn and wheel angular vel
    u_d = model.set_variable(var_type='_u', var_name='u_t')
    u_w = model.set_variable(var_type='_u', var_name='u_w')

    # Xy cg velocities
    vx = s_v * cos(s_b)
    vy = s_v * sin(s_b)

    # Wheel kinematics
    v_xf, v_yf = np.array([[cos(u_d), sin(u_d)], [-sin(u_d), cos(u_d)]]) @ np.array([vx, vy + l_f * s_r])
    v_xr, v_yr = vx, vy - l_r * s_r

    # Side slip angles
    a_f = -arctan(v_yf / abs(v_xf))
    a_r = -arctan(v_yr / abs(v_xr))

    # Slip ratios
    lam_f = 0
    lam_r = u_w * p - v_xr / np.maximum(abs(u_w * p), abs(v_xr))

    # Load forces
    F_zf = g * m * l_r / (l_f + l_r)
    F_zr = g * m * l_r / (l_f + l_r)

    # Tire models
    F_xf_raw = cDx * F_zf * sin(cCx * arctan(cBx * lam_f - cEx * (cBx * lam_f - arctan(cBx * lam_f))))
    F_xr_raw = cDx * F_zr * sin(cCx * arctan(cBx * lam_r - cEx * (cBx * lam_r - arctan(cBx * lam_r))))
    F_yf_raw = cDy * F_zf * sin(cCy * arctan(cBy * a_f - cEy * (cBy * a_f - arctan(cBy * a_f))))
    F_yr_raw = cDy * F_zr * sin(cCy * arctan(cBy * a_r - cEy * (cBy * a_r - arctan(cBy * a_r))))

    # Traction ellipse and scaling of force vector
    b_star = arccos(abs(lam_r + lam_f) / sqrt((lam_f + lam_r) ** 2 + sin(a_f + a_r) ** 2))

    # Front
    mu_xf_act = F_xf_raw / F_zf
    mu_yf_act = F_yf_raw / F_zf
    mu_xf_max = cDx
    mu_yf_max = cDy

    mu_xf = 1 / sqrt((1 / mu_xf_act) ** 2 + (1 / mu_yf_max) ** 2)
    mu_yf = tan(b_star) / sqrt((1 / mu_xf_max) ** 2 + (tan(b_star) / mu_yf_act) ** 2)

    F_xf = abs(mu_xf / mu_xf_act) * F_xf_raw
    F_yf = abs(mu_yf / mu_yf_act) * F_yf_raw

    # Rear
    mu_xr_act = F_xr_raw / F_zr
    mu_yr_act = F_yr_raw / F_zr
    mu_xr_max = cDx
    mu_yr_max = cDy

    mu_xr = 1 / sqrt((1 / mu_xr_act) ** 2 + (1 / mu_yr_max) ** 2)
    mu_yr = tan(b_star) / sqrt((1 / mu_xr_max) ** 2 + (tan(b_star) / mu_yr_act) ** 2)

    F_xr = abs(mu_xr / mu_xr_act) * F_xr_raw
    F_yr = abs(mu_yr / mu_yr_act) * F_yr_raw

    pm = np.array([[cos(u_d), -sin(u_d), 1, 0],
                   [sin(u_d), cos(u_d), 0, 1],
                   [l_f * cos(u_d), l_f * cos(u_d), 0, -l_r]])
    F_x, F_y, M_z = pm @ np.array([F_xf, F_yf, F_xr, F_yr])

    # Set right-hand-side of ODE for all introduced states (_x).
    m1 = np.array([[1 / m * s_v, 0, 0],
                   [0, 1 / m, 0],
                   [0, 0, 1/ I]])
    m2 = np.array([[-sin(s_b), cos(s_b), 0],
                   [cos(s_b), sin(s_b), 0],
                   [0, 0, 1]])
    m3 = np.array([[F_x],
                   [F_y],
                   [M_z]])

    main_rhs = m1 @ m2 @ m3

    model.set_rhs('s_b', main_rhs[0])
    model.set_rhs('s_v', main_rhs[1])
    model.set_rhs('s_r', main_rhs[2])
    model.set_rhs('s_x', s_v * cos(s_b))
    model.set_rhs('s_y', s_v * sin(s_b))
    model.set_rhs('s_phi', s_r)

    # Setup model:
    model.setup()

    return model