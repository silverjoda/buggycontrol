from abc import abstractmethod
import mujoco_py
import os
import numpy as np
import time

class Engine:
    def __init__(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def get_obs_dict(self):
        pass

    @abstractmethod
    def get_state_vec(self):
        pass

    @abstractmethod
    def get_complete_obs_vec(self):
        pass

    def set_trajectory(self):
        #self.model.body_pos[20]
        pass

    def movewaypoint(self, pos: np.ndarray):
        """
        move waypoint body to new position
        :param pos: waypoint position shape (2,)
        """
        self.mujoco_sim.data.set_mocap_pos(f"waypoint{self.firstwaypoint}", np.hstack((pos, [0])))
        self.firstwaypoint = (self.firstwaypoint + 1) % self.n_waypoints
        pass

    @abstractmethod
    def demo(self):
        pass


class MujocoEngine(Engine):
    def __init__(self, mujoco_sim):
        super().__init__()
        self.mujoco_sim = mujoco_sim
        self.model = mujoco_sim.model
        self.bodyid = self.model.body_name2id('buddy')

    def step(self, action):
        # Step simulation
        self.mujoco_sim.data.ctrl[:] = action
        self.mujoco_sim.forward()
        self.mujoco_sim.step()

    def reset(self):
        self.mujoco_sim.reset()

    def render(self):
        if not hasattr(self, 'viewer'):
            self.viewer = mujoco_py.MjViewerBasic(self.mujoco_sim)
        self.viewer.render()

    def get_obs_dict(self):
        pos = self.mujoco_sim.data.body_xpos[self.bodyid].copy()
        ori_q = self.mujoco_sim.data.body_xquat[self.bodyid].copy()
        ori_mat = self.mujoco_sim.data.body_xquat[self.bodyid].copy()
        vel = self.mujoco_sim.data.body_xvelp[self.bodyid].copy()
        ang_vel = self.mujoco_sim.data.body_xvelr[self.bodyid].copy()
        return {"pos" : pos, "ori_q" : ori_q, "ori_mat" : ori_mat, "vel" : vel, "ang_vel" : ang_vel}

    def get_state_vec(self):
        obs_dict = self.get_obs_dict()
        vel = obs_dict['vel']
        ang_vel = obs_dict['ang_vel']
        return [vel[0], vel[1], ang_vel[2]]

    def get_complete_obs_vec(self):
        # state+traj
        pass

    def demo(self):
        self.reset()
        while True:
            self.step([0.1,0.1])
            self.render()
            print(self.get_obs_dict())
            time.sleep(0.01)

class LTEEngine(Engine):
    def __init__(self, mujoco_sim, lte):
        super().__init__()
        self.mujoco_sim = mujoco_sim
        self.model = mujoco_sim.model
        self.bodyid = self.model.body_name2id('buddy')
        self.q_dim = self.mujoco_sim.get_state().qpos.shape[0]
        self.qvel_dim = self.mujoco_sim.get_state().qvel.shape[0]
        self.lte = lte
        self.dt = 0.01

        self.reset_vars()

    def step(self, action):
        # Make observation for lte out of current vel and action
        obs = *self.xy_vel, self.ang_vel_z[0], *action

        # Update velocities
        *self.xy_vel, self.ang_vel_z = self.lte.predict_next_vel(obs)

        # Update position
        self.xy_pos[0] += self.xy_vel[0] * self.dt
        self.xy_pos[1] += self.xy_vel[1] * self.dt

    def reset(self):
        self.reset_vars()

    def render(self):
        # Teleport mujoco model to given location
        qvel = np.zeros(self.qvel_dim)
        qpos = np.zeros(self.q_dim)
        qpos[0:1] = self.xy_pos
        qpos[3:7] = self.xy_pos
        old_state = self.mujoco_sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.mujoco_sim.set_state(new_state)
        self.mujoco_sim.forward()

        if not hasattr(self, 'viewer'):
            self.viewer = mujoco_py.MjViewerBasic(self.mujoco_sim)
        self.viewer.render()

    def theta_to_quat(self, theta):
        qx = 0
        qy = 0
        qz = np.sin(theta / 2)
        qw = np.cos(theta / 2)
        return [qw, qx, qy, qz]

    def reset_vars(self):
        self.xy_pos = [0, 0]
        self.xy_vel = [0, 0]
        self.theta = [0]
        self.ang_vel_z = [0]

    def get_obs_dict(self):
        pos = [*self.xy_pos, 0]
        ori_q = self.theta_to_quat(self.theta)
        ori_mat = np.eye(3)
        vel = [*self.xy_vel, 0]
        ang_vel = [0, 0, self.ang_vel_z]
        return {"pos": pos, "ori_q": ori_q, "ori_mat": ori_mat, "vel": vel, "ang_vel": ang_vel}

    def get_state_vec(self):
        obs_dict = self.get_obs_dict()
        vel = obs_dict['vel']
        ang_vel = obs_dict['ang_vel']
        return [vel[0], vel[1], ang_vel[2]]

    def get_complete_obs_vec(self):
        # state+traj
        pass

    def demo(self):
        self.reset()
        while True:
            self.step([0.1, 0.1])
            self.render()
            print(self.get_obs_dict())
            time.sleep(0.01)

if __name__ == "__main__":
    car_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/one_car.xml")
    model = mujoco_py.load_model_from_path(car_path)
    sim = mujoco_py.MjSim(model, nsubsteps=10)

    me = MujocoEngine(sim)
    me.demo()