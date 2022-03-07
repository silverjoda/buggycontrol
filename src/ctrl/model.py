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

    F_x_raw = F_xf_raw + F_xr_raw
    F_y_raw = F_yf_raw + F_yr_raw

    # Traction ellipse and scaling of force vector
    b_star = arccos(abs(lam_r + lam_f) / sqrt((lam_f + lam_r) ** 2 + sin(a_f + a_r) ** 2))

    mu_x_act = F_x_raw / (F_zf + F_zr)
    mu_y_act = F_y_raw / (F_zf + F_zr)

    mu_x_max = cDx
    mu_y_max = cDy

    mu_x = 1 / sqrt((1 / mu_x_act) ** 2 + (1 / mu_y_max) ** 2)
    mu_y = tan(b_star) / sqrt((1 / mu_x_max) ** 2 + (tan(b_star) / mu_y_act) ** 2)

    F_x = abs(mu_x / mu_x_act) * F_x_raw
    F_y = abs(mu_y / mu_y_act) * F_y_raw

    # Set right-hand-side of ODE for all introduced states (_x).
    model.set_rhs('C_b', ...)

    # Setup model:
    model.setup()

    return model