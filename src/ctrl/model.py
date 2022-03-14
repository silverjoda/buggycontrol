import do_mpc
from casadi import cos, sin, arctan, tan, sqrt, arccos, fabs, horzcat, vertcat, fmax, power, mpower


def make_singletrack_model(params=None):
    # Obtain an instance of the do-mpc model class
    # and select time discretization:
    model = do_mpc.model.Model('continuous')

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

    if params is not None:
        m, I, l_f, l_r, p, cDy, cBy, cCy, cEy, cDx, cBx, cCx, cEx = params

    # Introduce new states
    s_b = model.set_variable(var_type='_x', var_name='s_b', shape=(1,1))
    s_v = model.set_variable(var_type='_x', var_name='s_v', shape=(1,1))
    s_r = model.set_variable(var_type='_x', var_name='s_r', shape=(1,1))
    s_x = model.set_variable(var_type='_x', var_name='s_x', shape=(1,1))
    s_y = model.set_variable(var_type='_x', var_name='s_y', shape=(1,1))
    s_phi = model.set_variable(var_type='_x', var_name='s_phi', shape=(1,1))

    # Set control inputs to target turn and wheel angular vel
    u_d = model.set_variable(var_type='_u', var_name='u_d')
    u_w = model.set_variable(var_type='_u', var_name='u_w')

    # Xy cg velocities
    vx = s_v * cos(s_b)
    vy = s_v * sin(s_b)

    # Wheel kinematics
    v_f = vertcat(horzcat(cos(u_d), sin(u_d)), horzcat(-sin(u_d), cos(u_d))) @ vertcat(vx, vy + l_f * s_r)
    v_xf = v_f[0]
    v_yf = v_f[1]

    v_xr, v_yr = vx, vy - l_r * s_r

    # Side slip angles
    a_f = -arctan(v_yf / fabs(v_xf))
    a_r = -arctan(v_yr / fabs(v_xr))

    # Slip ratios
    lam_f = 0.
    lam_r = (u_w * p - v_xr) / fmax(fabs(u_w * p), fabs(v_xr))

    # Load forces
    F_zf = g * m * l_r / (l_f + l_r)
    F_zr = g * m * l_r / (l_f + l_r)

    # Tire models
    F_xf_raw = cDx * F_zf * sin(cCx * arctan(cBx * lam_f - cEx * (cBx * lam_f - arctan(cBx * lam_f))))
    F_xr_raw = cDx * F_zr * sin(cCx * arctan(cBx * lam_r - cEx * (cBx * lam_r - arctan(cBx * lam_r))))
    F_yf_raw = cDy * F_zf * sin(cCy * arctan(cBy * a_f - cEy * (cBy * a_f - arctan(cBy * a_f))))
    F_yr_raw = cDy * F_zr * sin(cCy * arctan(cBy * a_r - cEy * (cBy * a_r - arctan(cBy * a_r))))

    # Traction ellipse and scaling of force vector
    b_star_r = arccos(fabs(lam_r) / sqrt(power(lam_r, 2.) + power(sin(a_r), 2)))

    # Rear
    mu_xr_act = F_xr_raw / F_zr
    mu_yr_act = F_yr_raw / F_zr
    mu_xr_max = cDx
    mu_yr_max = cDy

    mu_xr = 1 / sqrt(power(1. / mu_xr_act, 2) + power(tan(b_star_r) / mu_yr_max, 2))
    mu_yr = tan(b_star_r) / sqrt(power(1 / mu_xr_max, 2) + power(tan(b_star_r) / mu_yr_act, 2))

    F_xf = F_xf_raw
    F_yf = F_yf_raw
    F_xr = fabs(mu_xr / mu_xr_act) * F_xr_raw
    F_yr = fabs(mu_yr / mu_yr_act) * F_yr_raw

    pm = vertcat(horzcat(cos(u_d), -sin(u_d), 1, 0),
                   horzcat(sin(u_d), cos(u_d), 0, 1),
                   horzcat(l_f * sin(u_d), l_f * cos(u_d), 0, -l_r))
    res = pm @ vertcat(F_xf, F_yf, F_xr, F_yr)
    F_x, F_y, M_z = res[0], res[1], res[2]

    # Set right-hand-side of ODE for all introduced states (_x).
    m1 = vertcat(horzcat(1 / m * s_v, 0, 0),
                   horzcat(0, 1 / m, 0),
                   horzcat(0, 0, 1 / I))
    m2 = vertcat(horzcat(-sin(s_b), cos(s_b), 0),
                 horzcat(cos(s_b), sin(s_b), 0),
                 horzcat(0, 0, 1))
    m3 = vertcat(F_x, F_y, M_z)

    main_rhs = m1 @ m2 @ m3

    model.set_rhs('s_b', main_rhs[0])
    model.set_rhs('s_v', main_rhs[1])
    model.set_rhs('s_r', main_rhs[2])
    model.set_rhs('s_x', s_v * cos(s_b))
    model.set_rhs('s_y', s_v * sin(s_b))
    model.set_rhs('s_phi', s_r)

    #model.set_variable(var_type='_tvp', var_name='trajectory_set_point_x')
    #model.set_variable(var_type='_tvp', var_name='trajectory_set_point_y')

    # Setup model:
    model.setup()

    return model


def make_bicycle_model(params=None):
    # Obtain an instance of the do-mpc model class
    # and select time discretization:
    model = do_mpc.model.Model('continuous')

    # Physical model:
    l_f = 0.164
    l_r = 0.160
    l_car = 0.535
    w_car = 0.281
    m_car = 4.0
    I_car = 0.12

    # Fitted tires parameters
    B_f = 29
    B_r = 26.9
    C_f = 0.0867
    C_r = 0.1632
    D_f = 42.52
    D_r = 161.58

    # Friction and motor parameters
    C_r0 = 0.6
    C_r2 = 0.1
    C_m1 = 1.8
    C_m2 = -0.25

    if params is not None:
        l_f, l_r, l_car, w_car, m_car, I_car, B_f, B_r, C_f, C_r, D_f, D_r, C_r0, C_r2, C_m1, C_m2 = params

    # Introduce new states
    s_x = model.set_variable(var_type='_x', var_name='s_x', shape=(1, 1))
    s_y = model.set_variable(var_type='_x', var_name='s_y', shape=(1, 1))
    s_phi = model.set_variable(var_type='_x', var_name='s_phi', shape=(1, 1))
    s_vx = model.set_variable(var_type='_x', var_name='s_vx', shape=(1,1))
    s_vy = model.set_variable(var_type='_x', var_name='s_vy', shape=(1,1))
    s_omega = model.set_variable(var_type='_x', var_name='s_omega', shape=(1,1))

    # Set control inputs to target turn and wheel angular vel
    u_d = model.set_variable(var_type='_u', var_name='u_d')
    u_D = model.set_variable(var_type='_u', var_name='u_D')

    a_f = -arctan((s_omega * l_f + s_vy) / s_vx) + u_d
    a_r = arctan((s_omega * l_r - s_vy) / s_vx)

    F_yf = D_f * sin(C_f * arctan(B_f * a_f))
    F_yr = D_r * sin(C_r * arctan(B_r * a_r))
    F_xr = (C_m1 - C_m2 * s_vx) * u_D - C_r2 * (s_vx ** 2) - C_r0

    model.set_rhs('s_x', s_vx * cos(s_phi) - s_vy * sin(s_phi))
    model.set_rhs('s_y', s_vx * sin(s_phi) + s_vy * cos(s_phi))
    model.set_rhs('s_phi', s_omega)
    model.set_rhs('s_vx', (1 / m_car) * (F_xr - F_yf * sin(u_d) + m_car * s_vy * s_omega))
    model.set_rhs('s_vy', (1 / m_car) * (F_yr - F_yf * cos(u_d) - m_car * s_vx * s_omega))
    model.set_rhs('s_omega', (1 / I_car) * (F_yf * l_f * cos(u_d) - F_yr * l_r))

    # Setup model:
    model.setup()

    return model