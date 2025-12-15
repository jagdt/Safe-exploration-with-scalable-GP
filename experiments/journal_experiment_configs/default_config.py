# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:09:14 2017

@author: tkoller
"""
import datetime
import warnings
from os import makedirs
from os.path import basename, splitext, dirname
from shutil import copy
from typing import Dict, Any


class DefaultConfig(object):
    """
    Options class for the exploration setting
    """
    # the verbosity level
    verbosity = 2
    ilqr_init = True

    type_perf_traj = 'taylor'
    r = 1
    perf_has_fb = True

    env_options = dict()

    def create_savedirs(self, file_path):
        """ """

        conf_name = splitext(basename(file_path))[0]

        self.save_path = None
        if self.save_results:
            if self.save_dir is None:
                time_string = datetime.datetime.now().strftime("%d-%m-%y-%H-%M-%S")
                res_path = "{}/res_{}_{}".format(self.save_path_base, conf_name, time_string)
            else:
                res_path = f"{self.save_path_base}/{self.save_dir}"

            try:
                makedirs(res_path)
            except Exception as e:
                warnings.warn(e)

            self.save_path = res_path
            # copy config file into results folder
            dirname_conf = dirname(file_path)
            copy("{}/{}.py".format(dirname_conf, conf_name), "{}/".format(res_path))

    def add_sacred_config(self, config: Dict[str, Any]) -> None:
        """Adds all the given pairs in the given config dict as attributes to this config object.

        :raises ValueError: if the given config key already exists
        """
        existing_keys = set(dir(self))
        for key, value in config.items():
            if key.startswith('_'):
                # This ignores things like doc strings.
                continue

            if key in existing_keys:
                raise ValueError(f'Param \'{key}\' already exists with value \'{value}\'')

            setattr(self, key, value)
