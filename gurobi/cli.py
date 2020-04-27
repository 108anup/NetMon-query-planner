from argparse import ArgumentParser, SUPPRESS

from solvers import solver_names


def generate_parser():
    p = ArgumentParser(
        description="Control",
        argument_default=SUPPRESS
    )

    p.add_argument(
        '-s', '--solver',
        action='store',
        help='Technique to use for placement',
        choices=solver_names
    )

    p.add_argument(
        '-i', '--input_num',
        action='store',
        type=int,
        help='Which input to use'
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
        '-r', '--results-file',
        action='store',
    )

    p.add_argument(
        '-c', '--config-file',
        action='append'
    )

    p.add_argument(
        '-p', '--prog-dir',
        action='store'
    )

    p.add_argument(
        '--init',
        action='store_true'
    )

    p.add_argument(
        '--parallel',
        action='store_true'
    )

    return p
