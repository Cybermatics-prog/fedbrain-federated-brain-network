# SPDX-License-Identifier: MIT
"""Single, cross-platform entry point for reproducing this project.

Every documented reproduction step is a subcommand here, so the same commands work on
Linux, macOS and Windows without a shell-specific wrapper:

    python scripts/reproduce.py setup      # create .venv and install requirements
    python scripts/reproduce.py data       # download the public ABIDE-I derivatives
    python scripts/reproduce.py features   # build the FC/BOLD cache used by the interface
    python scripts/reproduce.py demo       # train a federation trace the interface replays
    python scripts/reproduce.py test       # offline test suite
    python scripts/reproduce.py verify     # compliance, language, tests, data checks
    python scripts/reproduce.py serve      # start the API and interface
    python scripts/reproduce.py all        # data -> features -> demo -> test -> verify

The script never overwrites the recorded trace committed under
`runs/federation-demo-s42`; a reproduction run gets its own run id.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / '.venv'


def venv_python() -> Path:
    return VENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def python() -> Path:
    """Prefer the project environment, otherwise the interpreter running this script."""
    candidate = venv_python()
    if candidate.is_file() and candidate.resolve() != Path(sys.executable).resolve():
        return candidate
    return Path(sys.executable)


def display(part) -> str:
    """Show project files relative to the repository root in log output."""
    text = str(part)
    try:
        return str(Path(text).resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return text


def run(steps, dry_run: bool = False) -> None:
    for label, command, cwd in steps:
        print(f'\n== {label}\n   $ {" ".join(display(part) for part in command)}', flush=True)
        if dry_run:
            continue
        started = time.perf_counter()
        completed = subprocess.run([str(part) for part in command], cwd=str(cwd or ROOT))
        if completed.returncode != 0:
            raise SystemExit(f'step failed ({completed.returncode}): {label}')
        print(f'   ok in {time.perf_counter() - started:.1f}s', flush=True)


def step(label, *command, cwd=None):
    return (label, tuple(command), cwd)


def cmd_setup(dry_run: bool) -> None:
    # Create the environment first, then install into it: resolving the interpreter before
    # the venv exists would install the dependencies into whatever Python invoked this script.
    if not venv_python().is_file():
        run([step('create virtual environment', Path(sys.executable), '-m', 'venv', VENV)], dry_run)
    run([step('install dependencies', python(), '-m', 'pip', 'install', '-r',
              ROOT / 'requirements.txt')], dry_run)
    print('\nEnvironment ready. Next: python scripts/reproduce.py data')


def cmd_data(dry_run: bool) -> None:
    run([step('download ABIDE-I derivatives', python(), '-m', 'fedbrain.runner', 'download')], dry_run)


def cmd_features(dry_run: bool) -> None:
    run([step('build health-data cache', python(), ROOT / 'scripts' / 'build_health_data.py')], dry_run)


def demo_config() -> Path:
    """Use the reused-selection config only when its reference checkpoint exists."""
    reference = ROOT / 'runs' / 'full-fedbrain-s42' / 'checkpoint.pt'
    name = 'federation_demo.json' if reference.is_file() else 'federation_demo_standalone.json'
    return ROOT / 'configs' / name


def cmd_demo(run_id: str, dry_run: bool, force: bool) -> None:
    target = ROOT / 'runs' / run_id
    if target.exists() and not force:
        print(f'run directory {target.relative_to(ROOT)} already exists; leaving it untouched.\n'
              f'Pass --run-id <new-name> or --force to train again.')
        return
    config = demo_config()
    print(f'configuration: {config.relative_to(ROOT)}')
    if config.name == 'federation_demo_standalone.json':
        print('note: runs/full-fedbrain-s42/checkpoint.pt is absent, so the sparse selection is\n'
              '      fitted by this run. The trace has the same structure as the committed one but\n'
              '      different numeric values.')
    run([step('train federation trace', python(), '-m', 'fedbrain.runner', 'train',
              '--config', config, '--run-id', run_id)], dry_run)
    print(f'\nTrace written to runs/{run_id}. Inspect it with: python scripts/reproduce.py serve')


def cmd_test(dry_run: bool) -> None:
    run([step('offline tests', python(), '-m', 'unittest', 'discover', '-s', 'tests', '-q')], dry_run)


def cmd_verify(dry_run: bool) -> None:
    checks = [
        step('license and portability compliance', python(), ROOT / 'scripts' / 'check_compliance.py'),
        step('English-only repository', python(), ROOT / 'scripts' / 'check_no_chinese.py'),
        step('offline tests', python(), '-m', 'unittest', 'discover', '-s', 'tests', '-q'),
    ]
    if (ROOT / 'data' / 'abide' / 'manifest.json').is_file():
        checks.append(step('health views need no model checkpoint', python(),
                           ROOT / 'scripts' / 'check_health_independent.py'))
    else:
        print('note: data/abide is absent, skipping the health-data independence check')
    run(checks, dry_run)


def cmd_serve(port: int, dry_run: bool) -> None:
    run([step('serve API and interface', python(), '-m', 'fedbrain.server', '--port', str(port))], dry_run)


def cmd_all(args) -> None:
    cmd_data(args.dry_run)
    cmd_features(args.dry_run)
    cmd_demo(args.run_id, args.dry_run, args.force)
    cmd_test(args.dry_run)
    cmd_verify(args.dry_run)
    print('\nAll reproduction steps finished. Start the interface with:\n'
          '    python scripts/reproduce.py serve')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Reproduce the FedBrain six-site federated experiment and its interface.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='print the commands without running them')
    subparsers = parser.add_subparsers(dest='command', required=True)

    for name, help_text in (
        ('setup', 'create the virtual environment and install dependencies'),
        ('data', 'download the public ABIDE-I derivatives'),
        ('features', 'build the FC/BOLD cache used by the health views'),
        ('demo', 'train a federation trace that the interface replays'),
        ('test', 'run the offline test suite'),
        ('verify', 'run compliance, language, test and data checks'),
        ('all', 'run data, features, demo, test and verify'),
    ):
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        if name in ('demo', 'all'):
            sub.add_argument('--run-id', default='demo-reproduction',
                             help='run directory to write (default: demo-reproduction)')
            sub.add_argument('--force', action='store_true', help='retrain even if the run directory exists')
    serve = subparsers.add_parser('serve', help='start the API and interface')
    serve.add_argument('--port', type=int, default=8765, help='port to listen on (default: 8765)')

    args = parser.parse_args()
    handlers = {
        'setup': lambda: cmd_setup(args.dry_run),
        'data': lambda: cmd_data(args.dry_run),
        'features': lambda: cmd_features(args.dry_run),
        'test': lambda: cmd_test(args.dry_run),
        'verify': lambda: cmd_verify(args.dry_run),
        'serve': lambda: cmd_serve(args.port, args.dry_run),
        'all': lambda: cmd_all(args),
    }
    if args.command == 'demo':
        cmd_demo(args.run_id, args.dry_run, args.force)
    else:
        handlers[args.command]()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
