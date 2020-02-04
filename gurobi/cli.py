from argparse import ArgumentParser


def generate_parser():
    p = ArgumentParser(
        description="NetMon"
    )

    p.add_argument(
        '-s', '--scheme',
        action='store',
        default='netmon',
        help='Technique to use for placement',
        choices=['netmon', 'univmon', 'univmon_greedy', 'univmon_greedy_rows']
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
        default=0
    )

    return p
