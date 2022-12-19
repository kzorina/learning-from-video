from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt_utils.agents_nn import ModelPgNNContinuous
from rlpyt.utils.logging import logger
import numpy as np
import torch

class MinibatchRlADRWrapper(MinibatchRl):

    def __init__(self, rand_sampler=None, shared_dict=None, log_adr_steps=10,  log_diagnostics_fun=None, store_diagnostics_fun=None, **kwargs):
        super().__init__(**kwargs)
        self.rand_sampler = rand_sampler
        self.shared_dict = shared_dict
        self.log_adr_steps = log_adr_steps
        self.store_diagnostics_fun = store_diagnostics_fun
        self.log_diagnostics_fun = log_diagnostics_fun

    def store_diagnostics(self, itr, traj_infos, opt_info):
        super().store_diagnostics(itr, traj_infos, opt_info)
        if self.store_diagnostics_fun is not None:
            self.store_diagnostics_fun(itr, self.algo, self.agent, self.sampler)

    def log_diagnostics(self, itr):
        super().log_diagnostics(itr)
        if self.log_diagnostics_fun is not None:
            self.log_diagnostics_fun(itr, self.algo, self.agent, self.sampler)


    def check_adr(self, itr):
        if len(self.rand_sampler.params_to_randomize) > 0:
            self.rand_sampler.set_use_boundary(True)
            self.rand_sampler.prepare_check_adr()
            self.shared_dict['rand_sampler'] = self.rand_sampler
            # print("check ADR")
            self.agent.sample_mode(itr)
            samples, traj_infos = self.sampler.obtain_samples(itr)
            self.agent.train_mode(itr)
            print("check ADR")
            avg_return = np.mean([item['Return'] for item in traj_infos])
            print(avg_return)
            if np.isnan(avg_return):
                print('return was nan')
                print([item['Return'] for item in traj_infos])
                print(traj_infos)
            logger.record_tabular('ADR_avg_return', avg_return)
            print("rand i in check_Adr = {}".format(self.rand_sampler.i))
            current_snapshot_mode = logger.get_snapshot_mode()
            logger.set_snapshot_mode('all')
            updated = self.rand_sampler.update_bounds(avg_return)
            if updated:
                self.save_itr_snapshot(itr)
            logger.set_snapshot_mode(current_snapshot_mode)
            self.rand_sampler.set_use_boundary(False)
            self.shared_dict['rand_sampler'] = self.rand_sampler

    def log_adr(self, itr):
        for key, value in self.rand_sampler.get_parameters_named():
            logger.record_tabular(key, value)

    def train(self):
        """
        Performs startup, then loops by alternating between
        ``sampler.obtain_samples()`` and ``algo.optimize_agent()``, logging
        diagnostics at the specified interval.
        """
        n_itr = self.startup()
        for itr in range(n_itr):
            logger.set_iteration(itr)
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)  # Might not be this agent sampling.
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0 or itr == 0:
                    print('itr:', itr)
                    print('self.log_interval_itrs:', self.log_interval_itrs)
                    if (itr + 1) % self.log_adr_steps == 0:
                        # check ADR
                        self.check_adr(itr)
                        self.log_adr(itr)
                    self.log_diagnostics(itr)
        self.shutdown()

class FixedStdModel(ModelPgNNContinuous):
    def __init__(self, observation_shape, action_size, policy_hidden_sizes=None,
                 policy_hidden_nonlinearity=torch.nn.Tanh, value_hidden_sizes=None,
                 value_hidden_nonlinearity=torch.nn.Tanh, init_log_std=0., min_std=0., normalize_observation=False,
                 norm_obs_clip=10, norm_obs_var_clip=1e-6, pretrain_mu_file=None):
        super().__init__(observation_shape, action_size, policy_hidden_sizes, policy_hidden_nonlinearity,
                         value_hidden_sizes, value_hidden_nonlinearity, init_log_std, min_std, normalize_observation,
                         norm_obs_clip, norm_obs_var_clip, pretrain_mu_file)
        self._log_std.requires_grad = False

def log_diagnostics(itr, algo, agent, sampler):
    std = agent.model.log_std.exp().data.numpy()
    for i in range(std.shape[0]):
        logger.record_tabular('agent/std{}'.format(i), std[i])
    # record_tabular('agent/velocity_penalty_min', np.min(sampler.samples_np.env.env_info.velocity_violation_penalty))
    # record_tabular('agent/velocity_penalty_max', np.max(sampler.samples_np.env.env_info.velocity_violation_penalty))
