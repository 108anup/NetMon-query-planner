from argparse import ArgumentParser

from solvers import solver_names


def generate_parser():
    p = ArgumentParser(
        description="Control"
    )

    p.add_argument(
        '-s', '--solver',
        action='store',
        help='Technique to use for placement',
        choices=solver_names
    )

    p.add_argument(
        '-c', '--cfg_num',
        action='store',
        type=int,
        help='Which configuration to use'
    )

    p.add_argument(
        '-v', '--verbose',
        action='count',
    )

    p.add_argument(
        '--mipout',
        action='store_true',
    )

    p.add_argument(
        '--horizontal-partition', '--hp',
        action='store_true',
    )

    p.add_argument(
        '--vertical-partition', '--vp',
        action='store_true',
    )

    p.add_argument(
        '-o', '--output-file',
        action='store',
    )

    p.add_argument(
        '--config-file',
        action='append'
    )

    return p
