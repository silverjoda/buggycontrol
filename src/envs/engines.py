import os
from abc import abstractmethod
from copy import deepcopy

import mujoco_py
import numpy as np

from src.envs.xml_gen import *
from src.opt.simplex_noise import SimplexNoise
from src.policies import LTE, MLP
from src.utils import load_config, theta_to_quat, q2e, e2q

class Engine:
    def __init__(self, config):
        self.config = config
        car_xml = config["car_xml"]

        self.buddy_template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/cars/base_car/buddy.xml")
        self.buddy_rnd_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/cars/base_car/buddy_rnd.xml")

        self.car_template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"assets/models/{car_xml}.xml")
        self.car_rnd_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"assets/models/{car_xml}_rnd.xml")

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

    def set_trajectory(self, traj):
        self.wp_list = traj
        self.cur_wp_idx = 0
        self.cur_mujoco_wp_idx = 0
        self.set_wp_visuals()

    def reset_trajectory(self, traj_pts_in=None):
        self.wp_list = self.generate_random_traj(traj_pts_in=traj_pts_in)
        self.cur_wp_idx = 0
        self.cur_mujoco_wp_idx = 0
        self.set_wp_visuals()

    def set_wp_visuals(self):
        for i in range(self.config["n_traj_pts"]):
            self.mujoco_sim.data.set_mocap_pos(f"waypoint{i}", np.hstack((self.wp_list[i], [0])))

    def set_wp_visuals_externally(self, traj):
        assert len(traj) == self.config["n_traj_pts"]
        for i in range(self.config["n_traj_pts"]):
            self.mujoco_sim.data.set_mocap_pos(f"waypoint{i}", np.hstack((traj[i], [0])))

    def update_wp_visuals(self):
        self.mujoco_sim.data.set_mocap_pos(f"waypoint{self.cur_mujoco_wp_idx }", np.hstack((self.wp_list[self.cur_wp_idx + self.config["n_traj_pts"]], [0])))

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

    def get_state_vec(self, obs_dict):
        vel = obs_dict['vel']
        ang_vel = obs_dict['ang_vel']
        return [vel[0], vel[1], ang_vel[2]]

    def set_external_state(self, state_dict):
        old_state = self.mujoco_sim.get_state()
        qpos = old_state.qpos # qvel
        qpos[0:2] = state_dict["x_pos"], state_dict["y_pos"]
        quat = e2q(0,0,state_dict["phi"])
        qpos[3:7] = quat
        new_state = mujoco_py.MjSimState(old_state.time, qpos, old_state.qvel,
                                         old_state.act, old_state.udd_state)

        self.mujoco_sim.set_state(new_state)
        self.mujoco_sim.forward()

    def get_complete_obs_vec(self, allow_latent_input):
        obs_dict = self.get_obs_dict()
        state = self.get_state_vec(obs_dict)
        wps_buggy_frame = obs_dict["wp_list"]

        wps_contiguous = []
        for w in wps_buggy_frame:
            wps_contiguous.extend(w)

        complete_obs_vec = state + wps_contiguous
        if allow_latent_input:
            complete_obs_vec += self.random_params_normalized

        return complete_obs_vec, obs_dict

    def transform_wp_to_buggy_frame(self, wp_list, pos, ori_q):
        wp_arr = np.array(wp_list)
        wp_arr_centered = wp_arr - np.array(pos[0:2])
        buggy_q = ori_q
        _, _, theta = q2e(*buggy_q)
        t_mat = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
        wp_buggy = np.matmul(t_mat, wp_arr_centered.T).T
        return wp_buggy

    def transform_wp_to_buggy_frame_sar(self, wp_list, pos, ori_q):
        wp_arr = np.array(wp_list)
        wp_arr_centered = wp_arr - np.array(pos[0:2])
        buggy_q = ori_q
        _, _, theta = q2e(*buggy_q)
        t_mat = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
        wp_buggy = np.matmul(t_mat, wp_arr_centered.T).T

        X_new = np.zeros(len(wp_buggy))
        X_new[0] = np.arctan2(wp_buggy[0][1], wp_buggy[0][0])
        for i in range(1, len(wp_buggy)):
            X_new[i] = np.arctan2(wp_buggy[i][1] - wp_buggy[i - 1][1], wp_buggy[i][0] - wp_buggy[i - 1][0]) - X_new[i - 1]
        return X_new

    def generate_random_traj(self, traj_pts_in=None, plot=False):
        #traj_smoothness = self.config["traj_smoothness"] - self.current_difficulty * 200
        traj_smoothness = self.config["traj_smoothness"] - np.random.rand() * self.config["traj_smoothness_variance"]
        self.noise = SimplexNoise(dim=1, smoothness=traj_smoothness, multiplier=1)
        traj_pts = []
        current_xy = np.zeros(2)

        if traj_pts_in is None:
            # Generate fine grained trajectory
            for i in range(1300):
                noise = 5 * self.noise()[0]
                current_xy += np.array([0.01 * np.cos(noise), 0.01 * np.sin(noise)])
                traj_pts.append(deepcopy(current_xy))
        else:
            traj_pts = traj_pts_in

        # Sample equidistant points
        wp_list = []
        cum_wp_dist = 0
        for i in range(1, len(traj_pts)):
            cum_wp_dist += self.dist_between_wps(traj_pts[i], traj_pts[i-1])
            if cum_wp_dist > self.config["wp_sample_dist"]:
                wp_list.append(traj_pts[i])
                cum_wp_dist = 0

        # Debug plot
        if plot:
            self.plot_traj(traj_pts, wp_list)

        return wp_list

    def plot_traj(self, traj_pts, wp_list):
        # For debugging
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
    def __init__(self, config):
        super().__init__(config)
        self.model, self.mujoco_sim, self.bodyid = self.load_env()

    def generate_random_sim_parameters(self):
        # 0: friction (0.4), 1: steering_range (0.38), 2: body mass (3.47), 3: kv (3000), 4: gear (0.003)
        random_param_scale_offset_list = [[0.15, 0.4], [0.1, 0.38], [1, 3.5], [1000, 3000], [0.001, 0.003]]
        random_params_normalized = list(np.clip(np.random.randn(len(random_param_scale_offset_list)) * 0.4, -1, 1))
        random_params_sim = [(random_params_normalized[i] - rso[0]) / rso[1] for i, rso in enumerate(random_param_scale_offset_list)]
        return random_params_normalized, random_params_sim

    def load_env(self):
        if self.config["randomize_env"]:
            self.random_params_normalized, self.random_params_sim = self.generate_random_sim_parameters()

            buddy_xml = gen_buddy_xml(self.random_params_sim)
            with open(self.buddy_rnd_path, "w") as out_file:
                for s in buddy_xml.splitlines():
                    out_file.write(s)
                    out_file.write("\n")
            car_xml = gen_car_xml(self.random_params_sim)
            #model = mujoco_py.load_model_from_xml(car_xml)
            # This might be uneccessary to write the top level xml, as we can load it from xml directly and it will call the overwritten buddy xml
            with open(self.car_rnd_path, "w") as out_file:
                for s in car_xml.splitlines():
                    out_file.write(s)
                    out_file.write("\n")
            model = mujoco_py.load_model_from_path(self.car_rnd_path)
        else:
            model = mujoco_py.load_model_from_path(self.car_template_path)

        mujoco_sim = mujoco_py.MjSim(model, nsubsteps=self.config['n_substeps'])
        bodyid = model.body_name2id('buddy')

        return model, mujoco_sim, bodyid

    def step(self, action):
        # Step simulation
        self.mujoco_sim.data.ctrl[:] = action
        self.mujoco_sim.forward()
        self.mujoco_sim.step()

        return self.step_trajectory(self.mujoco_sim.data.body_xpos[self.bodyid].copy()[0:2])

    def reset(self):
        if self.config["randomize_env"] or not hasattr(self, "model"):
            self.model, self.mujoco_sim, self.bodyid = self.load_env()

        if hasattr(self, 'viewer'):
            del self.viewer

        self.reset_estimators()
        self.mujoco_sim.reset()
        self.reset_trajectory()

    def render(self):
        if not hasattr(self, 'viewer'):
            self.viewer = mujoco_py.MjViewer(self.mujoco_sim)
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
        turn_angle = 0#np.clip(self.mujoco_sim.get_state().qpos[7] * 2.5, -1, 1) #
        rear_wheel_speed = 0#np.clip(((self.mujoco_sim.get_state().qvel[14] + self.mujoco_sim.get_state().qvel[16]) / 2.) / 200, -1, 1)
        return {"pos" : pos, "ori_q" : ori_q, "ori_mat" : ori_mat, "vel" : vel_buggy, "ang_vel" : ang_vel, "wp_list" : wps_buggy_frame, "turn_angle" : turn_angle, "rear_wheel_speed" : rear_wheel_speed}

class LTEEngine(Engine):
    def __init__(self, config):
        super().__init__(config)
        self.lte = self.load_lte()
        self.model, self.mujoco_sim, self.bodyid = self.load_env()
        self.dt = 1. / self.config["rate"]
        self.reset_vars()

    def load_env(self):
        self.random_params_normalized = [0, 0, 0, 0, 0, 1.]
        self.random_params_sim = [0, 0, 0, 0, 0, 1.]

        # For visualization
        model = mujoco_py.load_model_from_path(self.car_template_path)
        mujoco_sim = mujoco_py.MjSim(model, nsubsteps=self.config['n_substeps'])
        bodyid = model.body_name2id('buddy')

        return model, mujoco_sim, bodyid

    def load_lte(self):
        import torch as T
        lte = LTE(obs_dim=self.config["state_dim"] + 2, act_dim=self.config["state_dim"], hid_dim=128)
        #lte = MLP(5, 3, hid_dim=128)
        lte_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "opt/agents/buggy_lte.p")
        lte.load_state_dict(T.load(lte_path), strict=False)
        return lte

    def step(self, action):
        # Make observation for lte out of current vel and action
        obs = np.array((*self.xy_vel, self.ang_vel_z, *action), dtype=np.float32)

        # Update velocities
        self.xy_vel[0], self.xy_vel[1], self.ang_vel_z = self.lte.predict_next_vel(obs)

        # Update position
        self.xy_pos[0] = self.xy_pos[0] + np.cos(self.theta) * self.xy_vel[0] * self.dt + np.sin(self.theta) * self.xy_vel[1] * self.dt
        self.xy_pos[1] = self.xy_pos[1] + np.sin(self.theta) * self.xy_vel[0] * self.dt + np.cos(self.theta) * self.xy_vel[1] * self.dt
        self.theta = self.theta + self.ang_vel_z * self.dt

        return self.step_trajectory(self.xy_pos)

    def reset(self):
        if not hasattr(self, "model"):
            self.model, self.mujoco_sim, self.bodyid = self.load_env()

        self.reset_vars()
        self.reset_trajectory()

    def render(self):
        if not hasattr(self, 'viewer'):
            self.viewer = mujoco_py.MjViewer(self.mujoco_sim)

        self.set_external_state({"x_pos": self.xy_pos[0],
                                 "y_pos": self.xy_pos[1],
                                 "phi": self.theta})

        self.viewer.render()

    def reset_vars(self):
        self.xy_pos = [0, 0]
        self.xy_vel = [0, 0]
        self.theta = 0
        self.ang_vel_z = 0

    def get_obs_dict(self):
        pos = [*self.xy_pos, 0]
        ori_q = theta_to_quat(self.theta)
        ori_mat = np.eye(3)
        vel = [*self.xy_vel, 0]
        ang_vel = [0, 0, self.ang_vel_z]
        wps = self.wp_list[self.cur_wp_idx:self.cur_wp_idx + self.config["n_traj_pts"]]
        wps_buggy_frame = self.transform_wp_to_buggy_frame(wps, pos, ori_q)
        return {"pos": pos, "ori_q": ori_q, "ori_mat": ori_mat, "vel": vel, "ang_vel": ang_vel, "wp_list": wps_buggy_frame}

class HybridEngine(Engine):
    def __init__(self, config):
        super().__init__(config)
        self.lte = self.load_lte()
        self.discriminator = self.load_discriminator()
        self.model, self.mujoco_sim, self.bodyid = self.load_env()
        self.dt = 1. / self.config["rate"]
        self.reset_vars()

    def load_env(self):
        self.random_params_normalized = [0, 0, 0, 0, 0, 1.]
        self.random_params_sim = [0, 0, 0, 0, 0, 1.]

        # For visualization
        model = mujoco_py.load_model_from_path(self.car_template_path)
        mujoco_sim = mujoco_py.MjSim(model, nsubsteps=self.config['n_substeps'])
        bodyid = model.body_name2id('buddy')

        return model, mujoco_sim, bodyid

    def load_lte(self):
        import torch as T
        lte = LTE(obs_dim=self.config["state_dim"] + 2, act_dim=self.config["state_dim"], hid_dim=128)
        #lte = MLP(5, 3, hid_dim=128)
        lte_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "opt/agents/buggy_lte.p")
        lte.load_state_dict(T.load(lte_path), strict=False)
        return lte

    def load_discriminator(self):
        import torch as T
        discriminator = MLP(5, 2, hid_dim=128)
        discriminator_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "opt/agents/data_discriminator.p")
        discriminator.load_state_dict(T.load(discriminator_path), strict=False)
        return discriminator

    def step(self, action):
        # Make observation for lte out of current vel and action
        obs = np.array((*self.xy_vel, self.ang_vel_z, *action), dtype=np.float32)

        # Decide if similar observation is present in the dataset
        pred = self.discriminator.predict_next(obs)
        use_real_policy = pred[1] > self.config["discriminator_thresh"]

        # Step mujoco
        self.mujoco_sim.data.ctrl[:] = action
        self.mujoco_sim.forward()
        self.mujoco_sim.step()

        # Step real policy and correct mujoco pos and vel
        if use_real_policy:
            # Update velocities
            self.xy_vel[0], self.xy_vel[1], self.ang_vel_z = self.lte.predict_next_vel(obs)

            # Update position
            self.xy_pos[0] = self.xy_pos[0] + np.cos(self.theta) * self.xy_vel[0] * self.dt + np.sin(self.theta) * self.xy_vel[1] * self.dt
            self.xy_pos[1] = self.xy_pos[1] + np.sin(self.theta) * self.xy_vel[0] * self.dt + np.cos(self.theta) * self.xy_vel[1] * self.dt
            self.theta = self.theta + self.ang_vel_z * self.dt

            # Update mujoco position and velocities
            old_state = self.mujoco_sim.get_state()
            qpos = old_state.qpos  # qvel
            qpos[0:2] = self.xy_pos[0], self.xy_pos[1]
            quat = e2q(0, 0, self.theta)
            qpos[3:7] = quat
            new_state = mujoco_py.MjSimState(old_state.time, qpos, old_state.qvel,
                                             old_state.act, old_state.udd_state)

            self.mujoco_sim.set_state(new_state)
            self.mujoco_sim.forward()

        else:
            # Get current vel, pos and ori from mujoco
            pos = self.mujoco_sim.data.body_xpos[self.bodyid].copy()
            ori_q = self.mujoco_sim.data.body_xquat[self.bodyid].copy()
            ori_mat = np.reshape(self.mujoco_sim.data.body_xmat[self.bodyid], (3, 3))
            vel_glob = self.mujoco_sim.data.body_xvelp[self.bodyid].copy()
            vel_buggy = np.matmul(ori_mat.T, vel_glob[:, np.newaxis])[:, 0]
            ang_vel = self.mujoco_sim.data.body_xvelr[self.bodyid].copy()

            # Set vel, pos and ori
            self.xy_vel[0], self.xy_vel[1], _ = vel_buggy
            self.ang_vel_z = ang_vel

            *self.xy_pos, _ = pos

            _, _, self.theta = q2e(*ori_q)

        return self.step_trajectory(self.xy_pos)

    def reset(self):
        if not hasattr(self, "model"):
            self.model, self.mujoco_sim, self.bodyid = self.load_env()

        self.reset_vars()
        self.reset_trajectory()

    def render(self):
        if not hasattr(self, 'viewer'):
            self.viewer = mujoco_py.MjViewer(self.mujoco_sim)
        self.viewer.render()

    def reset_vars(self):
        self.xy_pos = [0, 0]
        self.xy_vel = [0, 0]
        self.theta = 0
        self.ang_vel_z = 0

    def get_obs_dict(self):
        pos = [*self.xy_pos, 0]
        ori_q = theta_to_quat(self.theta)
        ori_mat = np.eye(3)
        vel = [*self.xy_vel, 0]
        ang_vel = [0, 0, self.ang_vel_z]
        wps = self.wp_list[self.cur_wp_idx:self.cur_wp_idx + self.config["n_traj_pts"]]
        wps_buggy_frame = self.transform_wp_to_buggy_frame(wps, pos, ori_q)
        return {"pos": pos, "ori_q": ori_q, "ori_mat": ori_mat, "vel": vel, "ang_vel": ang_vel, "wp_list": wps_buggy_frame}


if __name__ == "__main__":
    config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"))
    me = MujocoEngine(config)
    me.demo()