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
        choices=['netmon', 'univmon', 'univmon_greedy']
    )

    p.add_argument(
        '-c', '--config',
        action='store',
        default=0,
        help='Which configuration to use'
    )

    return p
