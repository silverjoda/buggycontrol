from abc import abstractmethod
import mujoco_py
import os
import numpy as np
import time
from src.opt.simplex_noise import SimplexNoise
from src.utils import load_config
from copy import deepcopy
import math as m

class Engine:
    def __init__(self, config, mujoco_sim):
        self.config = config
        self.mujoco_sim = mujoco_sim
        self.model = mujoco_sim.model
        self.current_difficulty = 0.

    def reset_trajectory(self):
        self.wp_list = self.generate_random_traj()
        self.cur_wp_idx = 0
        self.cur_mujoco_wp_idx = 0
        self.set_wp_visuals()

    def set_trajectory(self, traj):
        self.wp_list = traj
        self.cur_wp_idx = 0
        self.cur_mujoco_wp_idx = 0
        self.set_wp_visuals()

    def step_trajectory(self, pos):
        if self.cur_wp_idx >= len(self.wp_list):
            return True, False

        done = False
        wp_visited = False
        # Check if wp is reached
        if self.dist_between_wps(pos, self.wp_list[self.cur_wp_idx]) < self.config["wp_reach_dist"]:
            # Update wp visually in mujoco
            if self.config["render"]:
                self.update_wp_visuals()

            self.cur_wp_idx += 1
            self.cur_mujoco_wp_idx = (self.cur_mujoco_wp_idx + 1) % self.config["n_traj_pts"]

            wp_visited = True

        # Return done=true if we are at end of trajectory
        if self.cur_wp_idx == (len(self.wp_list) - self.config["n_traj_pts"]):
            done = True

        return done, wp_visited

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

    def get_state_vec(self, obs_dict):
        vel = obs_dict['vel']
        ang_vel = obs_dict['ang_vel']
        turn_angle = obs_dict['turn_angle']
        rear_wheel_speed = obs_dict['rear_wheel_speed']
        return [turn_angle, rear_wheel_speed, vel[0], vel[1], ang_vel[2]]

    def get_complete_obs_vec(self):
        exit()
        obs_dict = self.get_obs_dict()
        state = self.get_state_vec(obs_dict)
        wps_buggy_frame = obs_dict["wp_list"]

        wps_contiguous = []
        for w in wps_buggy_frame:
            wps_contiguous.extend(w)
        return state + wps_contiguous, obs_dict

    def transform_wp_to_buggy_frame(self, wp_list, pos, ori_q):
        wp_arr = np.array(wp_list)
        wp_arr_centered = wp_arr - np.array(pos[0:2])
        buggy_q = ori_q
        _, _, theta = self.q2e(*buggy_q)
        t_mat = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
        wp_buggy = np.matmul(t_mat, wp_arr_centered.T).T
        return wp_buggy

    def transform_wp_to_buggy_frame_sar(self, wp_list, pos, ori_q):
        wp_arr = np.array(wp_list)
        wp_arr_centered = wp_arr - np.array(pos[0:2])
        buggy_q = ori_q
        _, _, theta = self.q2e(*buggy_q)
        t_mat = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
        wp_buggy = np.matmul(t_mat, wp_arr_centered.T).T

        X_new = np.zeros(len(wp_buggy))
        X_new[0] = np.arctan2(wp_buggy[0][1], wp_buggy[0][0])
        for i in range(1, len(wp_buggy)):
            X_new[i] = np.arctan2(wp_buggy[i][1] - wp_buggy[i - 1][1], wp_buggy[i][0] - wp_buggy[i - 1][0]) - X_new[i - 1]
        return X_new

    def q2e(self, w, x, y, z):
        pitch = -m.asin(2.0 * (x * z - w * y))
        roll = m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
        yaw = m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
        return (roll, pitch, yaw)

    def generate_random_traj(self, traj_pts=None):
        #traj_smoothness = self.config["traj_smoothness"] - self.current_difficulty * 200
        traj_smoothness = self.config["traj_smoothness"] - np.random.rand() * self.config["traj_smoothness_variance"]
        self.noise = SimplexNoise(dim=1, smoothness=traj_smoothness, multiplier=1)
        traj_pts = []
        current_xy = np.zeros(2)

        # Generate fine grained trajectory
        for i in range(1300):
            noise = 5 * self.noise()[0]
            current_xy += np.array([0.01 * np.cos(noise), 0.01 * np.sin(noise)])
            traj_pts.append(deepcopy(current_xy))

        # Sample equidistant points
        wp_list = []
        cum_wp_dist = 0
        for i in range(1, len(traj_pts)):
            cum_wp_dist += self.dist_between_wps(traj_pts[i], traj_pts[i-1])
            if cum_wp_dist > self.config["wp_sample_dist"]:
                wp_list.append(traj_pts[i])
                cum_wp_dist = 0

        # Debug plot
        #self.plot_traj(traj_pts, wp_list)

        return wp_list

    def plot_traj(self, traj_pts, wp_list):
        traj_pts_x = [x for x,_ in traj_pts]
        traj_pts_y = [y for _, y in traj_pts]
        wp_x = [x for x,_ in wp_list]
        wp_y = [y for _, y in wp_list]
        import matplotlib.pyplot as plt
        plt.scatter(traj_pts_x, traj_pts_y)
        plt.show()
        exit()

    def dist_between_wps(self, wp_1, wp_2):
        return np.sqrt(np.square(wp_1[0] - wp_2[0]) + np.square(wp_1[1] - wp_2[1]))

    def set_wp_visuals(self):
        for i in range(self.config["n_traj_pts"]):
            self.mujoco_sim.data.set_mocap_pos(f"waypoint{i}", np.hstack((self.wp_list[i], [0])))

    def set_wp_visuals_externally(self, traj):
        assert len(traj) == self.config["n_traj_pts"]
        for i in range(self.config["n_traj_pts"]):
            self.mujoco_sim.data.set_mocap_pos(f"waypoint{i}", np.hstack((traj[i], [0])))

    def update_wp_visuals(self):
        self.mujoco_sim.data.set_mocap_pos(f"waypoint{self.cur_mujoco_wp_idx }", np.hstack((self.wp_list[self.cur_wp_idx + self.config["n_traj_pts"]], [0])))

    def demo(self):
        self.reset()
        while True:
            acts = [[0.0, 0.0], [0.0, 0.9], [-0.0, -0.9], [0.3, 0.9], [-0.3, 0.9]]
            for a in acts:
                for i in range(100):
                    self.step(a)
                    self.render()
                    #print(self.get_obs_dict()["vel"])
                    #time.sleep(0.001)

                    #print(self.get_state_vec())

                    q_vel_list = self.mujoco_sim.get_state().qvel[0:3]
                    q_mat = np.reshape(self.mujoco_sim.data.body_xmat[self.bodyid], (3, 3))
                    q_vel_buggy = np.matmul(np.linalg.inv(q_mat), q_vel_list[:, np.newaxis])

                    #print(q_vel_buggy[0:2, 0])
                    #print(self.mujoco_sim.data.body_xvelr[self.bodyid].copy()[2])
                    #print(self.cur_wp_idx, self.cur_mujoco_wp_idx)

class MujocoEngine(Engine):
    def __init__(self, config, mujoco_sim):
        super().__init__(config, mujoco_sim)
        self.bodyid = self.model.body_name2id('buddy')

    def step(self, action):
        # Step simulation
        self.mujoco_sim.data.ctrl[:] = action
        self.mujoco_sim.forward()
        self.mujoco_sim.step()

        # Update hidden states
        #self.update_estimators(action)

        return self.step_trajectory(self.mujoco_sim.data.body_xpos[self.bodyid].copy()[0:2])

    def reset(self):
        self.reset_estimators()
        self.mujoco_sim.reset()
        self.reset_trajectory()

    def render(self):
        if not hasattr(self, 'viewer'):
            self.viewer = mujoco_py.MjViewerBasic(self.mujoco_sim)
        self.viewer.render()

    def update_estimators(self, act):
        # turn: -0.4,0.4, throttle: 0,1
        turn, throttle = act
        if turn >= 0:
            pass
            #self.turn_est = np.clip(self.turn_est + turn, -0.4, np.minimum(0.4))

    def reset_estimators(self):
        self.turn_est = 0
        self.thrttle_est = 0

    def get_obs_dict(self):
        pos = self.mujoco_sim.data.body_xpos[self.bodyid].copy()
        ori_q = self.mujoco_sim.data.body_xquat[self.bodyid].copy()
        ori_mat = np.reshape(self.mujoco_sim.data.body_xmat[self.bodyid], (3, 3))
        vel_glob = self.mujoco_sim.data.body_xvelp[self.bodyid].copy()
        vel_buggy = np.matmul(ori_mat.T, vel_glob[:, np.newaxis])[:, 0]
        ang_vel = self.mujoco_sim.data.body_xvelr[self.bodyid].copy()
        wps = self.wp_list[self.cur_wp_idx:self.cur_wp_idx + self.config["n_traj_pts"]]
        wps_buggy_frame = self.transform_wp_to_buggy_frame(wps, pos, ori_q)
        turn_angle = np.clip(self.mujoco_sim.get_state().qpos[7] * 2.5, -1, 1)
        rear_wheel_speed = np.clip(((self.mujoco_sim.get_state().qvel[14] + self.mujoco_sim.get_state().qvel[16]) / 2.) / 200, -1, 1)
        return {"pos" : pos, "ori_q" : ori_q, "ori_mat" : ori_mat, "vel" : vel_buggy, "ang_vel" : ang_vel, "wp_list" : wps_buggy_frame, "turn_angle" : turn_angle, "rear_wheel_speed" : rear_wheel_speed}

    def get_complete_obs_vec(self):
        obs_dict = self.get_obs_dict()
        state = self.get_state_vec(obs_dict)
        wps_buggy_frame = obs_dict["wp_list"]

        wps_contiguous = []
        for w in wps_buggy_frame:
            wps_contiguous.extend(w)
        return state + wps_contiguous, obs_dict

class MujocoEngine2(Engine):
    def __init__(self, config, mujoco_sim):
        super().__init__(config, mujoco_sim)
        self.bodyid = self.model.body_name2id('buddy')

    def step(self, action):
        # Step simulation
        self.mujoco_sim.data.ctrl[:] = action
        self.mujoco_sim.forward()
        self.mujoco_sim.step()

        # Update hidden states
        #self.update_estimators(action)

        return self.step_trajectory(self.mujoco_sim.data.body_xpos[self.bodyid].copy()[0:2])

    def reset(self):
        self.reset_estimators()
        self.mujoco_sim.reset()
        self.reset_trajectory()

    def render(self):
        if not hasattr(self, 'viewer'):
            self.viewer = mujoco_py.MjViewerBasic(self.mujoco_sim)
        self.viewer.render()

    def update_estimators(self, act):
        # turn: -0.4,0.4, throttle: 0,1
        turn, throttle = act
        if turn >= 0:
            pass
            #self.turn_est = np.clip(self.turn_est + turn, -0.4, np.minimum(0.4))

    def reset_estimators(self):
        self.turn_est = 0
        self.throttle_est = 0

    def get_obs_dict(self):
        pos = self.mujoco_sim.data.body_xpos[self.bodyid].copy()
        ori_q = self.mujoco_sim.data.body_xquat[self.bodyid].copy()
        ori_mat = np.reshape(self.mujoco_sim.data.body_xmat[self.bodyid], (3, 3))
        vel_glob = self.mujoco_sim.data.body_xvelp[self.bodyid].copy()
        vel_buggy = np.matmul(ori_mat.T, vel_glob[:, np.newaxis])[:, 0]
        ang_vel = self.mujoco_sim.data.body_xvelr[self.bodyid].copy()
        wps = self.wp_list[self.cur_wp_idx:self.cur_wp_idx + self.config["n_traj_pts"]]
        wps_buggy_frame = self.transform_wp_to_buggy_frame_sar(wps, pos, ori_q)
        turn_angle = np.clip(self.mujoco_sim.get_state().qpos[7] * 2.5, -1, 1)
        rear_wheel_speed = np.clip(((self.mujoco_sim.get_state().qvel[14] + self.mujoco_sim.get_state().qvel[16]) / 2.) / 200, -1, 1)
        return {"pos" : pos, "ori_q" : ori_q, "ori_mat" : ori_mat, "vel" : vel_buggy, "ang_vel" : ang_vel, "wp_list" : wps_buggy_frame, "turn_angle" : turn_angle, "rear_wheel_speed" : rear_wheel_speed}

    def get_complete_obs_vec(self):
        obs_dict = self.get_obs_dict()
        state = self.get_state_vec(obs_dict)
        wps_buggy_frame = obs_dict["wp_list"]

        return state + list(wps_buggy_frame), obs_dict

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
        obs = *self.xy_vel, self.ang_vel_z, *action

        # Update velocities
        *self.xy_vel, self.ang_vel_z = self.lte.predict_next_vel(obs)

        # Update position
        self.xy_pos[0] += self.xy_vel[0] * self.dt
        self.xy_pos[1] += self.xy_vel[1] * self.dt
        self.theta += self.ang_vel_z * self.dt

        return self.step_trajectory(self.xy_pos)

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
        self.theta = 0
        self.ang_vel_z = 0

    def get_obs_dict(self):
        pos = [*self.xy_pos, 0]
        ori_q = self.theta_to_quat(self.theta)
        ori_mat = np.eye(3)
        vel = [*self.xy_vel, 0]
        ang_vel = [0, 0, self.ang_vel_z]
        wps = self.wp_list[self.cur_wp_idx:self.cur_wp_idx + self.config["n_traj_pts"]]
        return {"pos": pos, "ori_q": ori_q, "ori_mat": ori_mat, "vel": vel, "ang_vel": ang_vel, "wp_list": wps}

if __name__ == "__main__":
    config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"))
    car_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/one_car.xml")
    model = mujoco_py.load_model_from_path(car_path)
    sim = mujoco_py.MjSim(model, nsubsteps=config["n_substeps"])

    me = MujocoEngine(config, sim)
    me.demo()