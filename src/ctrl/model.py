import do_mpc

def make_model():
    # Obtain an instance of the do-mpc model class
    # and select time discretization:
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # Introduce new states, inputs and other variables to the model, e.g.:
    C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1,1))
    ...

    Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')
    ...

    # Set right-hand-side of ODE for all introduced states (_x).
    # Names are inherited from the state definition.
    model.set_rhs('C_b', ...)

    # Setup model:
    model.setup()

    return model