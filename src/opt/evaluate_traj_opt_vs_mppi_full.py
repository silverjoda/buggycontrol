from stable_baselines3 import A2C

from src.ctrl.control_buggy_mppi import ControlBuggyMPPI
from src.envs.buggy_env_mujoco import BuggyEnv
from src.utils import *
from src.policies import *
from stable_baselines3 import A2C

from src.ctrl.control_buggy_mppi import ControlBuggyMPPI
from src.envs.buggy_env_mujoco import BuggyEnv
from src.policies import *
from src.utils import *


class BuggyControlTester:
    def __init__(self):
        # Make buggy env
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"), 'r') as f:
            self.buggy_config = yaml.load(f, Loader=yaml.FullLoader)
        self.buggy_maize_env = BuggyEnv(self.buggy_config)

        # Make Mppi algo
        with open(os.path.join(os.path.dirname(__file__), "configs/control_buggy_mppi.yaml"), 'r') as f:
            self.mppi_config = yaml.load(f, Loader=yaml.FullLoader)
        self.mppi_algo = ControlBuggyMPPI(self.mppi_config, self.buggy_config)

        # Load RL agent(s)
        with open(os.path.join(os.path.dirname(__file__), "configs/train_buggy_a2c.yaml"), 'r') as f:
            self.algo_config = yaml.load(f, Loader=yaml.FullLoader)
        self.buggy_rl_policy = A2C.load("agents/{}_SB_policy".format(self.algo_config["session_ID"]))

        # Load tep(s)
        self.tep = TEPMLP(obs_dim=50, act_dim=1)
        self.tep.load_state_dict(T.load("agents/full_traj_tep.p"), strict=False)

    def eval_control_algos(self):
        # Reset env and record trajectory
        self.buggy_maize_env.reset()
        test_traj = self.buggy_maize_env.engine.wp_list

        # Test buggy agent on default shortest path
        default_rl_agent_rew, default_rl_agent_time_taken = self.test_rl_agent(self.buggy_maize_env, test_traj)

        # Update the shortest path using tep
        updated_traj = self.tep.optimize_traj(test_traj)

        # Test buggy agent on new path
        updated_traj_rl_agent_rew, updated_rl_agent_time_taken = self.test_rl_agent(self.buggy_maize_env, updated_traj)

        # Test mppi
        mppi_traj_follower_rew, mppi_traj_follower_time_taken = self.eval_mppi(self.buggy_maize_env, test_traj, follow_traj=True)
        mppi_free_rew, mppi_free_time_taken = self.eval_mppi(self.buggy_maize_env, test_traj, follow_traj=True)


    def test_rl_agent(self, env, test_traj):
        pass

    def eval_mppi(self, env, mppi_algo, follow_traj=True):
        pass