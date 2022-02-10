from abc import abstractmethod
import mujoco_py
import os
import numpy as np
import time
from src.opt.simplex_noise import SimplexNoise
from src.utils import load_config

class Engine:
    def __init__(self, config, mujoco_sim):
        self.config = config
        self.mujoco_sim = mujoco_sim
        self.model = mujoco_sim.model

    def reset_trajectory(self):
        self.wp_list = self.generate_random_traj()
        self.cur_wp_idx = 0
        self.cur_mujoco_wp_idx = 0
        self.set_wp_visuals()

    def step_trajectory(self, pos):
        # Check if wp is reached
        if self.dist_between_wps(pos, self.wp_list[self.cur_wp_idx]) < self.config["wp_reach_dist"]:
            self.cur_wp_idx += 1
            self.cur_mujoco_wp_idx = (self.cur_mujoco_wp_idx + 1) % self.config["n_traj_pts"]

            # Update wp visually in mujoco
            self.update_wp_visuals()

        # Return done=true if we are at end of trajectory
        if self.cur_wp_idx == (len(self.wp_list) - self.config["n_traj_pts"]):
            return True
        return False

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

    def get_state_vec(self):
        obs_dict = self.get_obs_dict()
        vel = obs_dict['vel']
        ang_vel = obs_dict['ang_vel']
        return [vel[0], vel[1], ang_vel[2]]

    def get_complete_obs_vec(self):
        state = self.get_state_vec()
        wps = self.wp_list[self.cur_wp_idx:self.cur_wp_idx + self.config["n_traj_pts"]]
        wps_contiguous = []
        for w in wps:
            wps_contiguous.extend(w)
        return state + wps

    def generate_random_traj(self):
        self.noise = SimplexNoise(dim=2, smoothness=100, multiplier=0.1)
        traj_pts = []
        current_xy = np.zeros(2)

        # Generate fine grained trajectory
        for i in range(1000):
            current_xy += self.noise()
            traj_pts.append(current_xy)

        # Sample equidistant points
        way_pts = []
        current_wp = [0, 0]
        for t in traj_pts:
            if self.dist_between_wps(t, current_wp) > self.config["wp_sample_dist"]:
                current_wp = t
                way_pts.append(t)

        return way_pts

    def dist_between_wps(self, wp_1, wp_2):
        return np.sqrt(np.square(wp_1[0] - wp_2[0]) + np.square(wp_1[1] - wp_2[1]))

    def set_wp_visuals(self):
        for i in range(self.config["n_traj_pts"]):
            self.mujoco_sim.data.set_mocap_pos(f"waypoint{i}", np.hstack((self.wp_list[i], [0])))

    def update_wp_visuals(self):
        self.mujoco_sim.data.set_mocap_pos(f"waypoint{self.cur_mujoco_wp_idx}", np.hstack((self.wp_list[self.cur_wp_idx + 1], [0])))

    def demo(self):
        self.reset()
        while True:
            self.step([0.1, 0.1])
            self.render()
            print(self.get_obs_dict())
            time.sleep(0.01)

class MujocoEngine(Engine):
    def __init__(self, config, mujoco_sim):
        super().__init__(config, mujoco_sim)
        self.bodyid = self.model.body_name2id('buddy')

    def step(self, action):
        # Step simulation
        self.mujoco_sim.data.ctrl[:] = action
        self.mujoco_sim.forward()
        self.mujoco_sim.step()

        self.step_trajectory(self.mujoco_sim.data.body_xpos[self.bodyid].copy()[0:2])

    def reset(self):
        self.mujoco_sim.reset()
        self.reset_trajectory()

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
        wps = self.wp_list[self.cur_wp_idx:self.cur_wp_idx + self.config["n_traj_pts"]]
        return {"pos" : pos, "ori_q" : ori_q, "ori_mat" : ori_mat, "vel" : vel, "ang_vel" : ang_vel, "wp_list" : wps}


class LTEEngine(Engine):
    def __init__(self, config, mujoco_sim, lte):
        super().__init__(config, mujoco_sim)
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
        self.theta += self.ang_vel_z * self.dt

        self.step_trajectory(self.xy_pos)

    def reset(self):
        self.reset_vars()
        self.reset_trajectory()

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

if __name__ == "__main__":
    car_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/one_car.xml")
    model = mujoco_py.load_model_from_path(car_path)
    sim = mujoco_py.MjSim(model, nsubsteps=10)

    config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"))
    me = MujocoEngine(config, sim)
    me.demo()