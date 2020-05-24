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
    MIP_GAP_REL=0.05,
    MIP_GAP_ABS_RES=20,
    MAX_DEVICES_PER_CLUSTER=25,
    MAX_CLUSTERS_PER_CLUSTER=200,
    ME_WEIGHT=5,
    CPU_CORE_WEIGTH=10,
    WORKERS=4,
    PORTION_TIME_ON_PERF=0.7,
    ABS_TIME_ON_UNIVMON_BOTTLENECK=None,
    MIP_GAP_REL_UNIVMON_BOTTLENECK=0.2,

    solver='Netmon',
    input_num=0,
    verbose=0,
    mipout=False,
    horizontal_partition=False,
    vertical_partition=False,
    results_file=None,
    output_file=None,
    config_file=[],
    prog_dir=None,
    init=False,
    parallel=False,
    perf_obj=False
)
common_config = Config()
common_config.update(default_config)
common_config.load_config_file()
