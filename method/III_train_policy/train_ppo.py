from method.III_train_policy.ppo_utils import MinibatchRlADRWrapper, FixedStdModel, log_diagnostics
from method.III_train_policy.training_sampler import RandParamSamplerADR
from method.utils import default_robot_params_dict, get_default_robot_z
from rlpyt_utils.agents_nn import AgentPgContinuous, ModelPgNNContinuous
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from pyphysx_envs.envs import RobotEnv
from pyphysx_utils.urdf_robot_parser import quat_from_euler
from rlpyt_utils import args
from multiprocessing import Manager
import torch
import numpy as np
import pickle
import yaml

manager = Manager()
shared_dict = manager.dict()

parser = args.get_default_rl_parser()
args.add_default_ppo_args(parser)
parser.add_argument('--obs_remove_time', dest='obs_add_time', action='store_false')
parser.add_argument('--add_dense_reward', dest='add_dense_reward', action='store_true')
parser.add_argument('--add_manual_shaped_reward', dest='add_manual_shaped_reward', action='store_true')
parser.add_argument('--baseline_no_reward', dest='baseline_no_reward', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('-yaml', type=str, default="train_ppo.yaml", help='Yaml file with expement params.')
parser.add_argument('-alignment_path', type=str, default="align_spade_1.pkl", help='Path to the alignment file.')
parser.add_argument('-ddp_q_path', type=str, default="q_traj_spade_1.pkl", help='Path to the ddp q data.')
parser.add_argument('-pretrain_path', type=str, default="pretrain_mu_spade_1.pkl", help='Path to the pretrained network data.')
parser.add_argument('-seed', type=int, default="0", help='seed')
parser.add_argument('-tool', type=str, default='spade', help='Tool name')
parser.add_argument('-robot', type=str, default='panda', help='Robot name')
parser.add_argument('-fps', type=int, default=24, help='Video fps')
parser.add_argument('-threads', type=int, default=35, help='Video fps')
options = parser.parse_args()
yaml_params = yaml.load(open(options.yaml), Loader=yaml.FullLoader)
d = vars(options)
d.update(yaml_params)

# load alignment, q_traj
alignment_parameters = pickle.load(open(options.alignment_path, 'rb'))
ddp_q = pickle.load(open(options.ddp_q_path, 'rb'))

x_base, y_base = ddp_q[-1][:2]
base_rot = ddp_q[-1][2]
q_trajectory_sampled = np.array([x[3:] for x in ddp_q])
robot_z_pose = get_default_robot_z(options.robot, options.tool)
horizon = len(q_trajectory_sampled)

is_manual_reward_run = options.add_dense_reward or options.add_manual_shaped_reward or options.baseline_no_reward
sampler_cls = SerialSampler if args.is_evaluation(options) else CpuSampler

rand_sampler = RandParamSamplerADR(alignment_parameters,
                                   params_to_randomize=options.adr['params_to_randomize'],
                                   delta=options.adr['delta'],
                                   thresholds_high=options.adr['thresholds_high'],
                                   thresholds_low=options.adr['thresholds_low'],
                                   # experiment_bounds=[[-0.16, 0.23], [-0.7, -0.1], [0., 0.]]
                                   )
shared_dict['rand_sampler'] = rand_sampler
default_robot_params = default_robot_params_dict(options.tool)
sampler = sampler_cls(
    # TODO: create class getter
    EnvCls=RobotEnv,
    env_kwargs=dict(
        **default_robot_params,
        scene_name=options.tool,
        tool_name=options.tool,
        robot_name=options.robot,
        render=args.is_evaluation(options),
        rate=options.fps,

        params=alignment_parameters,
        demonstration_q=q_trajectory_sampled,
        robot_pose=((x_base, y_base, robot_z_pose), quat_from_euler('xyz', [0., 0., base_rot])),

        add_dense_reward=options.add_dense_reward,
        add_manual_shaped_reward=options.add_manual_shaped_reward,
        scene_demo_importance=0,
        shared_dict=shared_dict,
        batch_T=horizon,

        render_dict=dict(
            use_meshcat=True,
            open_meshcat=True,
            wait_for_open=True,
            render_to_animation=True,
            animation_fps=24,
            # show_frames=True,
        )
    ),

    batch_T=horizon if args.is_evaluation(options) else horizon * 2,
    batch_B=1 if args.is_evaluation(options) else options.threads,

    max_decorrelation_steps=0,
)

agent = AgentPgContinuous(
    options.greedy_eval,
    initial_model_state_dict=args.load_initial_model_state(options),
    ModelCls=ModelPgNNContinuous if is_manual_reward_run else FixedStdModel,
    model_kwargs=dict(
        policy_hidden_sizes=options.policy['policy_hidden_sizes'],
        policy_hidden_nonlinearity=torch.nn.Tanh,
        value_hidden_sizes=options.policy['value_hidden_sizes'],
        value_hidden_nonlinearity=torch.nn.Tanh,
        min_std=options.policy['min_std'],
        init_log_std=np.log(options.policy['init_std']),
        pretrain_mu_file=None if is_manual_reward_run else options.pretrain_path
    )
)

runner = MinibatchRlADRWrapper(
    algo=args.get_ppo_from_options(options), agent=agent, sampler=sampler,
    n_steps=int(10000 * sampler.batch_size), log_interval_steps=int(10 * sampler.batch_size),
    affinity=args.get_affinity(options),
    log_adr_steps=100,
    shared_dict=shared_dict,
    rand_sampler=rand_sampler,
    log_diagnostics_fun=log_diagnostics,
    seed=options.seed,
)

with args.get_default_context(options, snapshot_mode='gap', snapshot_gap=100):
    runner.train()