from argparse import ArgumentParser

from solvers import solver_names


def generate_parser():
    p = ArgumentParser(
        description="NetMon"
    )

    p.add_argument(
        '-s', '--scheme',
        action='store',
        default='netmon',
        help='Technique to use for placement',
        choices=solver_names
    )

    p.add_argument(
        '-c', '--config',
        action='store',
        default=0,
        help='Which configuration to use'
    )

    p.add_argument(
        '-v', '--verbose',
        action='count',
        default=0
    )

    p.add_argument(
        '--mipout',
        action='store_true',
        default=False
    )

    p.add_argument(
        '--horizontal-partition', '--hp',
        action='store_true',
        default=False
    )

    p.add_argument(
        '-o', '--output-file',
        action='store',
        default=None
    )

    p.add_argument(
        '--vertical-partition', '--vp',
        action='store_true',
        default=False
    )

    return p
