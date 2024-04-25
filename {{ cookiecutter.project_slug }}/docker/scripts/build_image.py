# pylint: disable=missing-docstring
import subprocess
from pathlib import Path


def main():
    dockerfile = Path(__file__).parents[1] / 'Dockerfile'
    subprocess.run(
        [
            'docker',
            'build',
            '-t',
            'projeto',
            '--build-arg="USERNAME=user"',
            '-f',
            str(dockerfile),
            '.',
        ],
        check=False,
    )


if __name__ == '__main__':
    main()
