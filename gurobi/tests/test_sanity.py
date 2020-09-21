import math
import pytest
from input import beluga20, tofino, Input, del0
from devices import CPU, P4
from sketches import cm_sketch
from flows import flow
from config import common_config
from common import Namespace
import tests.utilities as ut
from tests.utilities import (run_all_with_input, setup_test_meta, combinations,
                             flatten_list)

ut.base_dir = 'outputs/sanity'


@pytest.mark.parametrize(
    "sketches, col_bits",
    # combinations([[1, 2, 3, 4], [10, 16, 17, 18, 20]])
    # [(16, 15)]
    [(18, 8)]
)
def test_CPU_P4_cm(sketches, col_bits):
    cols = 2**col_bits
    eps = math.e / cols
    inp = Input(
        devices=[
            CPU(**beluga20, name='CPU_1'),
            P4(**tofino, name='P4_1'),
        ],
        queries=[
            cm_sketch(eps0=eps, del0=del0)
            for i in range(sketches)
        ],
        flows=[
            flow(path=(0, 1), queries=[(i, 1)])
            for i in range(sketches)
        ]
    )

    common_config.vertical_partition = True
    # common_config.horizontal_partition = True

    m = Namespace()
    m.test_name = 'CPU_P4_cm'
    m.args_str = "sketches={};col_bits={}".format(sketches, col_bits)
    setup_test_meta(m)
    run_all_with_input(m, inp)


@pytest.mark.parametrize(
    "sketch_col_bits",
    [[17]*3 + [16]*4 + [15]*4]
)
def test_CPU_P4_different_sized_sketches(sketch_col_bits):
    sketch_cols = [2**col_bits for col_bits in sketch_col_bits]
    sketch_eps = [math.e / c for c in sketch_cols]
    inp = Input(
        devices=[
            CPU(**beluga20, name='CPU_1'),
            P4(**tofino, name='P4_1'),
        ],
        queries=[
            cm_sketch(eps0=eps, del0=del0)
            for eps in sketch_eps
        ],
        flows=[
            flow(path=(0, 1), queries=[(i, 1)])
            for i in range(len(sketch_cols))
        ]
    )

    common_config.vertical_partition = True
    # common_config.horizontal_partition = True

    m = Namespace()
    m.test_name = 'CPU_P4_different_sized_sketches'
    m.args_str = "sketch_col_bits={}".format(sketch_col_bits)
    setup_test_meta(m)
    run_all_with_input(m, inp)
