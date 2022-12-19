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


def evaluate(model_params_path, alignment_path, ddp_q_path, pretrain_path,
             seed=0, tool='spade', robot='panda', fps=24,
             yaml_file="train_ppo.yaml", visualize=False, verbose=False):
    manager = Manager()
    shared_dict = manager.dict()

    parser = args.get_default_rl_parser()
    args.add_default_ppo_args(parser)
    # options = parser.parse_args()
    options, _ = parser.parse_known_args()
    yaml_params = yaml.load(open(yaml_file), Loader=yaml.FullLoader)
    d = vars(options)
    d.update(yaml_params)

    # load alignment, q_traj
    alignment_parameters = pickle.load(open(alignment_path, 'rb'))
    ddp_q = pickle.load(open(ddp_q_path, 'rb'))

    x_base, y_base = ddp_q[-1][:2]
    base_rot = ddp_q[-1][2]
    q_trajectory_sampled = np.array([x[3:] for x in ddp_q])
    robot_z_pose = get_default_robot_z(robot, tool)
    horizon = len(q_trajectory_sampled)
    sampler_cls = SerialSampler

    rand_sampler = RandParamSamplerADR(alignment_parameters,
                                       params_to_randomize=options.adr['params_to_randomize'],
                                       delta=options.adr['delta'],
                                       thresholds_high=options.adr['thresholds_high'],
                                       thresholds_low=options.adr['thresholds_low'],
                                       # experiment_bounds=[[-0.16, 0.23], [-0.7, -0.1], [0., 0.]]
                                       )
    shared_dict['rand_sampler'] = rand_sampler
    default_robot_params = default_robot_params_dict(tool)
    sampler = sampler_cls(
        # TODO: create class getter
        EnvCls=RobotEnv,
        env_kwargs=dict(
            **default_robot_params,
            scene_name=tool,
            tool_name=tool,
            robot_name=robot,
            render=visualize,
            rate=fps,
            params=alignment_parameters,
            demonstration_q=q_trajectory_sampled,
            robot_pose=((x_base, y_base, robot_z_pose), quat_from_euler('xyz', [0., 0., base_rot])),
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
        batch_T=horizon,
        batch_B=1,
        max_decorrelation_steps=0,
    )

    initial_model_state_dict = None
    if model_params_path is not None:
        data = torch.load(model_params_path)
        initial_model_state_dict = data['agent_state_dict']
    agent = AgentPgContinuous(
        greedy_mode=True,
        initial_model_state_dict=initial_model_state_dict,
        ModelCls=FixedStdModel,
        model_kwargs=dict(
            policy_hidden_sizes=options.policy['policy_hidden_sizes'],
            policy_hidden_nonlinearity=torch.nn.Tanh,
            value_hidden_sizes=options.policy['value_hidden_sizes'],
            value_hidden_nonlinearity=torch.nn.Tanh,
            min_std=options.policy['min_std'],
            init_log_std=np.log(options.policy['init_std']),
            pretrain_mu_file=pretrain_path
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
        seed=seed,
    )


    runner.startup()
    # while True:
    sampler.obtain_samples(0)
    r = sampler.samples_np.env.reward
    if verbose:
        print(r)
    return r.sum()

import pathlib

if __name__ == "__main__":

    # 1) To run pretrained policies (init_policy)

    # tool_name = 'hammer'
    # video_id = 1
    # seeds = [123, 42, 77331, 321, 234]
    # pretrain_dir = f"{pathlib.Path(__file__).parent.parent.parent}/data/alignment/debug_hammer_1"
    # alignment_path = pathlib.PurePath(pretrain_dir, f"align_{tool_name}_{video_id}.pkl")
    # ddp_q_path = pathlib.PurePath(pretrain_dir, f"q_traj_{tool_name}_{video_id}.pkl")
    # pretrained_mu_path = pathlib.PurePath(pretrain_dir, f"pretrained_mu_panda_{tool_name}_{video_id}.pkl")
    # model_params_path = pathlib.PurePath(pretrain_dir, f"itr_99.pkl")
    # evaluate(str(model_params_path), str(alignment_path), str(ddp_q_path), str(pretrained_mu_path),
    #          seed=seeds[1], tool=tool_name, verbose=True, visualize=True)

    # for tool_name in ['spade']:
    #     for video_id in range(5, 6):
    #         pretrain_dir = f"{pathlib.Path(__file__).parent.parent.parent}/data/alignment/pretrained/panda/{tool_name}/video_{video_id}"
    #         alignment_path = pathlib.PurePath(pretrain_dir, f"align_{tool_name}_{video_id}.pkl")
    #         ddp_q_path = pathlib.PurePath(pretrain_dir, f"q_traj_{tool_name}_{video_id}.pkl")
    #         pretrained_mu_path = pathlib.PurePath(pretrain_dir, f"pretrained_mu_panda_{tool_name}_{video_id}.pkl")
    #
    #         evaluate(None, str(alignment_path), str(ddp_q_path), str(pretrained_mu_path),
    #                  seed=123, tool=tool_name, verbose=True, visualize=True)

    # 2) To run the final policy
    # for tool_name in ['hammer']:
    #     for video_id in range(2, 6):
    #         pretrain_dir = f"{pathlib.Path(__file__).parent.parent.parent}/data/alignment/pretrained/panda/{tool_name}/video_{video_id}"
    #         final_policy_dir = f"{pathlib.Path(__file__).parent.parent.parent}/data/pretrain_components/final_policy/{tool_name}/video_{video_id}"
    #         alignment_path = pathlib.PurePath(pretrain_dir, f"align_{tool_name}_{video_id}.pkl")
    #         ddp_q_path = pathlib.PurePath(pretrain_dir, f"q_traj_{tool_name}_{video_id}.pkl")
    #         pretrained_mu_path = pathlib.PurePath(pretrain_dir, f"pretrained_mu_panda_{tool_name}_{video_id}.pkl")
    #         model_params_path = pathlib.PurePath(final_policy_dir, 'policy.pkl')
    #         evaluate(str(model_params_path), str(alignment_path), str(ddp_q_path), str(pretrained_mu_path),
    #                  seed=123, tool=tool_name, verbose=True)
    tool_name = 'spade'
    video_id = 1
    test_next_itr = ['itr_0.pkl', 'itr_99.pkl']
    for itr_name in test_next_itr:
        pretrain_dir = f"{pathlib.Path(__file__).parent.parent.parent}/data/alignment/debug_spade_1"
        alignment_path = pathlib.PurePath(pretrain_dir, f"align_{tool_name}_{video_id}.pkl")
        ddp_q_path = pathlib.PurePath(pretrain_dir, f"q_traj_{tool_name}_{video_id}.pkl")
        pretrained_mu_path = pathlib.PurePath(pretrain_dir, f"pretrained_mu_panda_{tool_name}_{video_id}.pkl")
        model_params_path = pathlib.PurePath(pretrain_dir, itr_name)
        evaluate(str(model_params_path), str(alignment_path), str(ddp_q_path), str(pretrained_mu_path),
                 seed=123, tool=tool_name, verbose=True, visualize=True)
