
# 0: friction (0.4), 1: steering_range (0.38), 2: body mass (3.47), 3: kv (3000), 4: gear (0.003)
def gen_buddy_xml(params):
    buddy_str = """<mujoco>
      <compiler meshdir="cars/meshes" />
    
      <asset>
        <mesh name="buddy_mushr_base_nano" file="mushr_base_nano.stl"/>
        <mesh name="buddy_mushr_wheel" file="mushr_wheel.stl"/>
        <mesh name="buddy_mushr_ydlidar" file="mushr_ydlidar.stl"/>
      </asset>
    
      <default>
          <default class="buddy_wheel_collision">
            <geom type="ellipsoid"  size="0.05 0.03 0.05"  friction="{} 0.005 0.0001" condim="3" contype="1" conaffinity="0" mass="0.498952" solimp="0 0.95 0.001 0.4 2" solref="0.02 1" margin="0"/>
          </default> 
    
          <default class="buddy_wheel_visual">
            <geom type="ellipsoid"  size="0.09 0.09 0.2"  friction="0.6 0.005 0.0001" condim="3" contype="1" conaffinity="0" mesh="buddy_mushr_wheel" mass="0.498952"/>
          </default>
    
          <default class="buddy_steering">
            <joint type="hinge" axis="0 0 1" limited="true" frictionloss="0.01" damping="0.001" armature="0.0002" range="-{} {}"/>
          </default>
    
          <default class="buddy_throttle">
            <joint type="hinge" axis="0 1 0" frictionloss="0.000" damping="0.00" armature="0.00" limited="false"/>
          </default>
    
          <default class="buddy_suspension">
            <joint type="slide" axis="0 0 1" frictionloss="0.001" stiffness="500.0" springref="-0.05" damping="12.4" armature="0.01" limited="false"/>
          </default>
      </default>
    
      <worldbody>
        <body name="buddy" pos="0.0 0.0 0.0" euler="0 0 0.0">
          <camera name="buddy_third_person" mode="fixed" pos="-1 0 1" xyaxes="0 -1 0 0.707 0 0.707"/>
          <joint type="free"/>
    
          <camera name="buddy_realsense_d435i" mode="fixed" pos="-0.005 0 .165" euler="0 4.712 4.712"/>
          <site name="buddy_imu" pos="-0.005 0 .165"/>
    
          <geom pos="0 0 0.094655" type="mesh" mass="{}" mesh="buddy_mushr_base_nano"/>
          <geom name="buddy_realsense_d435i" size="0.012525 0.045 0.0125" pos="0.0123949 0 0.162178" mass="0.072" type="box"/>
          <geom name="buddy_ydlidar" pos="-0.035325 0 0.202405" type="mesh" mass="0.180" mesh="buddy_mushr_ydlidar"/>
    
          <body name="buddy_steering_wheel" pos="0.1385 0 0.0488">
            <joint class="buddy_steering" name="buddy_steering_wheel"/>
            <geom class="buddy_wheel_collision" contype="0" conaffinity="0" mass="0.01" rgba="0 0 0 0.01"/>
          </body>
    
          <body name="buddy_wheel_fl" pos="0.1385 0.115 0.0488">
            <joint class="buddy_suspension" name="buddy_wheel_fl_suspension"/>
            <joint class="buddy_steering" name="buddy_wheel_fl_steering"/>
            <joint class="buddy_throttle" name="buddy_wheel_fl_throttle"/>
            <geom class="buddy_wheel_collision"/>
            <geom class="buddy_wheel_visual" type="mesh" contype="0" conaffinity="0" group="1" rgba="1 1 1 0.3"/>
          </body>
          <body name="buddy_wheel_fr" pos="0.1385 -0.115 0.0488">
          <joint class="buddy_suspension" name="buddy_wheel_fr_suspension"/>
            <joint class="buddy_steering" name="buddy_wheel_fr_steering"/>
            <joint class="buddy_throttle" name="buddy_wheel_fr_throttle"/>
            <geom class="buddy_wheel_collision"/>
            <geom class="buddy_wheel_visual" type="mesh" contype="0" conaffinity="0" group="1" rgba="1 1 1 0.3"/>
          </body>
          <body name="buddy_wheel_bl" pos="-0.158 0.115 0.0488">
            <joint class="buddy_suspension" name="buddy_wheel_bl_suspension"/>
            <joint class="buddy_throttle" name="buddy_wheel_bl_throttle"/>
            <geom class="buddy_wheel_collision"/>
            <geom class="buddy_wheel_visual" type="mesh" contype="0" conaffinity="0" group="1" rgba="1 1 1 0.3"/>
          </body>
          <body name="buddy_wheel_br" pos="-0.158 -0.115 0.0488">
            <joint class="buddy_suspension" name="buddy_wheel_br_suspension"/>
            <joint class="buddy_throttle" name="buddy_wheel_br_throttle"/>
            <geom class="buddy_wheel_collision"/>
            <geom class="buddy_wheel_visual" type="mesh" contype="0" conaffinity="0" group="1" rgba="1 1 1 0.3"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position class="buddy_steering" kp="25.0" name="buddy_steering_pos" joint="buddy_steering_wheel"/>
        <velocity kv="3000" gear="0.003" forcelimited="true" forcerange="-2000 2000" name="buddy_throttle_velocity" tendon="buddy_throttle"/>
      </actuator>
      <equality>
        <!-- taylor expansion of delta_l = arctan(L/(L/tan(delta) - W/2)) with L,W in reference to kinematic car model -->
        <joint joint1="buddy_wheel_fl_steering" joint2="buddy_steering_wheel" polycoef="0 1 0.375 0.140625 -0.0722656"/>
    
        <!-- taylor expansion of delta_r = arctan(L/(L/tan(delta) + W/2)) with L,W in reference to kinematic car model -->
        <joint joint1="buddy_wheel_fr_steering" joint2="buddy_steering_wheel" polycoef="0 1 -0.375 0.140625 0.0722656"/>
      </equality>
      <tendon>
        <fixed name="buddy_throttle">
          <joint joint="buddy_wheel_fl_throttle" coef="0"/>
          <joint joint="buddy_wheel_fr_throttle" coef="0"/>
          <joint joint="buddy_wheel_bl_throttle" coef="0.4"/>
          <joint joint="buddy_wheel_br_throttle" coef="0.4"/>
        </fixed>
      </tendon>
      <sensor>
        <accelerometer name="buddy_accelerometer" site="buddy_imu" />
        <gyro name="buddy_gyro" site="buddy_imu" />
        <velocimeter name="buddy_velocimeter" site="buddy_imu" />
      </sensor>
    </mujoco>""".format(*params[:5])
    return buddy_str


def gen_car_xml(params):
    car_str = """<mujoco model="mushr_nano">
      <compiler angle="radian" />
      <size njmax="500" nconmax="300"/>
      <option timestep="0.001" integrator="RK4" solver="Newton" o_solimp="0 0.95 0.001 0.4 2" o_solref="0.02 1" o_margin="0"/>
      <include file="cars/base_car/buddy.xml"/>
      <asset>
        <texture name="texplane" type="2d" builtin="checker" rgb1="0.26 0.12 0.36" rgb2="0.23 0.09 0.33" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
        <texture name="texgeom" type="cube" builtin="flat" mark="cross" width="127" height="1278" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>
        <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
        <material name="matgeom" texture="texgeom" texuniform="true" rgba="0.8 0.6 .4 1"/>
      </asset>
      <visual>
        <headlight ambient="0.6 0.6 0.6" diffuse="0.4 0.4 0.4" specular="0.2 0.2 0.2"/>
        <map znear="0.001" />
      </visual>
      <worldbody>
        <geom contype="1" name="floor" friction="{} 0.005 0.0001" pos="0 0 0" size="0 0 .25" type="plane" material="matplane" condim="3" solimp="0 0.95 0.001 0.4 2" solref="0.02 1" margin="0"/>
            <body name="waypoint0" pos="-10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05" type="sphere" mass="1" rgba="1 0 0 1"/>
            </body>
            <body name="waypoint1" pos="-10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05" type="sphere" mass="1" rgba="1 0 0 1"/>
            </body>
            <body name="waypoint2" pos="-10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05" type="sphere" mass="1" rgba="1 0 0 1"/>
            </body>
            <body name="waypoint3" pos="-10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05" type="sphere" mass="1" rgba="1 0 0 1"/>
            </body>
            <body name="waypoint4" pos="-10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05" type="sphere" mass="1" rgba="1 0 0 1"/>
            </body>
            <body name="waypoint5" pos="-10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05" type="sphere" mass="1" rgba="1 0 0 1"/>
            </body>
            <body name="waypoint6" pos="-10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05" type="sphere" mass="1" rgba="1 0 0 1"/>
            </body>
            <body name="waypoint7" pos="-10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05" type="sphere" mass="1" rgba="1 0 0 1"/>
            </body>
            <body name="waypoint8" pos="-10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05" type="sphere" mass="1" rgba="1 0 0 1"/>
            </body>
            <body name="waypoint9" pos="-10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05" type="sphere" mass="1" rgba="1 0 0 1"/>
            </body>
            <body name="waypoint10" pos="-10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05" type="sphere" mass="1" rgba="1 0 0 1"/>
            </body>
            <body name="waypoint11" pos="-10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05" type="sphere" mass="1" rgba="1 0 0 1"/>
            </body>
            <body name="waypoint12" pos="-10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05" type="sphere" mass="1" rgba="1 0 0 1"/>
            </body>
            <body name="waypoint13" pos="-10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05" type="sphere" mass="1" rgba="1 0 0 1"/>
            </body>
            <body name="waypoint14" pos="-10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05" type="sphere" mass="1" rgba="1 0 0 1"/>
            </body>
                <body name="barrier1" pos="10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05 0.5 .2" type="box" mass="1" rgba="0 0 1 1"/>
            </body>
            <body name="barrier2" pos="10 0 0" mocap="true">
            <geom contype="0" conaffinity="0" condim="3" size=".05 0.5 .2" type="box" mass="1" rgba="0 0 1 1"/>
            </body>
      </worldbody>
    </mujoco> """.format(params[0])
    return car_str
