import os
import yaml

from common import Namespace


class Config(Namespace):

    def load_config_file(self, fpath='config.yml'):
        if(os.path.exists(fpath)):
            f = open(fpath)
            config_data = yaml.safe_load(f)
            f.close()
            self.update(config_data)

    def update(self, config):
        if(isinstance(config, dict)):
            self.__dict__.update(config)
        else:
            self.__dict__.update(config.__dict__)


'''
Config Priorities:
cli input overrides
cli provided config file overrides
default config file overrides
default config values
'''

default_config = Config(
    time_limit=None,
    tolerance=0.999,
    ns_tol=0,
    res_tol=0,
    use_model=False,
    ftol=6.0e-5,
    mipgapabs=10,
    # mipgap=0.01,

    solver='Netmon',
    input_num=0,
    verbose=0,
    mipout=False,
    horizontal_partition=False,
    vertical_partition=False,
    results_file=None,
    config_file=[],
    prog_dir=None,
    init=False
)
common_config = Config()
common_config.update(default_config)
common_config.load_config_file()
