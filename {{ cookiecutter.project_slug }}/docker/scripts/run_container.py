"""A script to run our Docker Data Science container.

This script instantiates a Docker Data Science container with .

Usage:
    $ python xyz_script.py [options]

Options:
    -h, --help          Show this help message and exit.
    -i, --input FILE    Input file containing data (default: stdin).
    -o, --output FILE   Output file to write results (default: stdout).
    -v, --verbose       Increase verbosity level.

Examples:
    $ python xyz_script.py -i input.txt -o output.txt
    This command processes data from input.txt and writes results to output.txt.

    $ python xyz_script.py -v
    This command runs the script in verbose mode, providing more detailed output.

Author: Your Name <your.email@example.com>
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args()


if __name__ == "__main__":
    main()
