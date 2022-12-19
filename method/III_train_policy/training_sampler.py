#!/usr/bin/env python

# Copyright (c) CTU  - All Rights Reserved
# Created on: 1/28/20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

import copy
import numpy as np
from random import sample


def rand_init_pose(init_pose, rand_state):
    pos, quat = init_pose
    return pos + rand_state.normal(0., 0.05), quat

#
# def rand_param_sampler(init_params, rand_state):
#     params = copy.deepcopy(init_params)
#     # params['goal_box_position'][:2] += rand_state.uniform(-0.5, 0.5, size=2)
#     # params['goal_box_position'][0] += rand_state.uniform(-0.3, 0.0, size=1)
#     params['spheres_mass'] = rand_state.uniform(0.1, 0.3)
#     return params


class RandParamSamplerADR:
    # todo: add buffer (?)

    def __init__(self, init_params, params_to_randomize, delta, min_max_bounds=None, experiment_bounds=None,
                 thresholds_low=0.15, thresholds_high=1.2):
        # todo: add check for same length of each parameter
        self.init_params = {key: init_params[key] for key in params_to_randomize}
        self.params_to_randomize = params_to_randomize
        if min_max_bounds is None:
            min_max_bounds = {}
            for key in params_to_randomize:
                min_max_bounds[key] = []
                for i, item in enumerate(delta[key]):
                    min_max_bounds[key].append([self.init_params[key][i] - 10*item, self.init_params[key][i] + 10*item])
        self.params, self.min_max_bounds, self.delta = self.params_dict_to_array(params_to_randomize,
                                                                                 init_params, min_max_bounds, delta)
        if experiment_bounds is None:
            self.bounds = [[x, x] for x in self.params]
            print('bounds: ', self.bounds)
        else:
            self.bounds = experiment_bounds
        self.i = None
        self.thresholds_low = thresholds_low
        self.thresholds_high = thresholds_high
        self.use_boundary = False
        self.use_up_boundary = None
        self.fixed_parameters = None

    def __call__(self, *args, **kwargs):
        self.params = self.sample_parameters()
        params, bounds_arr, _ = self.params_array_to_dict()
        return params, bounds_arr

    def params_dict_to_array(self, params_to_randomize, params, bounds, delta):
        params_arr = []
        bounds_arr = []
        delta_arr = []
        for key in params_to_randomize:
            params_arr += list(params[key])
            bounds_arr += list(bounds[key])
            delta_arr += list(delta[key])
        return params_arr, bounds_arr, delta_arr

    def params_array_to_dict(self):
        start_len = 0
        params_dict = {}
        bounds_dict = {}
        delta_dict = {}
        for key in self.params_to_randomize:
            len_add = len(self.init_params[key])
            params_dict[key] = self.params[start_len:start_len + len_add]
            bounds_dict[key] = self.bounds[start_len:start_len + len_add]
            delta_dict[key] = self.delta[start_len:start_len + len_add]
            start_len += len_add
        return params_dict, bounds_dict, delta_dict

    def set_use_boundary(self, use_boundary):
        self.use_boundary = use_boundary

    def set_fixed_parameters(self):
        self.fixed_parameters = self.sample_parameters()

    def sample_lambda(self):
        return [np.random.uniform(low_bound, up_bound, size=1)[0] for low_bound, up_bound in self.bounds]


    def prepare_check_adr(self):
        x = np.random.uniform(0, 1, size=1)
        self.i = sample(range(0, len(self.bounds)), 1)[0]
        if x < 0.5:
            self.use_up_boundary = False
        else:
            self.use_up_boundary = True

    def sample_parameters(self):
        lambda_par = self.sample_lambda()
        if self.use_boundary:
            if self.use_up_boundary:
                lambda_par[self.i] = self.bounds[self.i][1]
            else:
                lambda_par[self.i] = self.bounds[self.i][0]
        return lambda_par

    def update_bounds(self, metric):
        if self.i is not None:
            if metric > self.thresholds_high:
                # check if we still in observation boundary
                if self.use_up_boundary:
                    self.bounds[self.i][1] = min(self.bounds[self.i][1] + self.delta[self.i],
                                                 self.min_max_bounds[self.i][1])
                else:
                    self.bounds[self.i][0] = max(self.bounds[self.i][0] - self.delta[self.i],
                                                 self.min_max_bounds[self.i][0])
                return True
            elif metric < self.thresholds_low:
                # check if changing boundary is viable
                if self.use_up_boundary:
                    self.bounds[self.i][1] = max(self.bounds[self.i][1] - self.delta[self.i], self.bounds[self.i][0])
                else:
                    self.bounds[self.i][0] = min(self.bounds[self.i][0] + self.delta[self.i], self.bounds[self.i][1])
        return False


    def get_parameters(self):
        return [item for sublist in self.bounds for item in sublist]

    def get_parameters_named(self):
        ident_bound = ['low', 'up']
        params_dict, bounds_dict, delta_dict = self.params_array_to_dict()
        return [("{}/{}_{}".format(key, i, ident_bound[j]), item)
                for key, value in bounds_dict.items()
                for i, bounds in enumerate(value)
                for j, item in enumerate(bounds)]
