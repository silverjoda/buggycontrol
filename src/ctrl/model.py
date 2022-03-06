import do_mpc

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
    c_t = model.set_variable(var_type='_u', var_name='c_t')
    c_w = model.set_variable(var_type='_u', var_name='c_w')


    # Set right-hand-side of ODE for all introduced states (_x).
    # Names are inherited from the state definition.
    model.set_rhs('C_b', ...)

    # Setup model:
    model.setup()

    return model