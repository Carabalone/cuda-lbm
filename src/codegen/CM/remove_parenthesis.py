import re
import sys
import argparse


def add_brackets(text):
    pattern = re.compile(r'k(\d+)')
    return pattern.sub(lambda m: f'k[{m.group(1)}]', text)


def remove_brackets(text):
    pattern = re.compile(r'k\[(\d+)\]')
    return pattern.sub(lambda m: f'k{m.group(1)}', text)

def offset_indices(text, delta):
    pattern = re.compile(r'k(?:\[(\d+)\]|(\d+))')

    def repl(m: re.Match) -> str:
        orig = m.group(1) or m.group(2)
        new_i = int(orig) + delta
        if m.group(1) is not None:
            # was bracketed
            return f'k[{new_i}]'
        else:
            # was plain
            return f'k{new_i}'

    return pattern.sub(repl, text)


def main():
    p = argparse.ArgumentParser(
        description="Add/remove brackets around k<digits>/k[digits] and "
                    "optionally offset each index by DELTA."
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        '-a', '--add',
        action='store_true',
        help="convert k123 → k[123]"
    )
    mode.add_argument(
        '-r', '--remove',
        action='store_true',
        help="convert k[123] → k123"
    )
    p.add_argument(
        '-d', '--delta',
        type=int,
        help="integer offset to add to each index (can be negative)"
    )
    p.add_argument(
        'infile',
        type=argparse.FileType('r'),
        help="input file"
    )
    p.add_argument(
        '-o', '--outfile',
        type=argparse.FileType('w'),
        default=sys.stdout,
        help="output file (defaults to stdout)"
    )
    args = p.parse_args()

    text = args.infile.read()
    if args.add:
        text = add_brackets(text)
    elif args.remove:
        text = remove_brackets(text)

    if args.delta is not None:
        text = offset_indices(text, args.delta)

    args.outfile.write(text)


if __name__ == "__main__":
    main()