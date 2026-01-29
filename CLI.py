#!/usr/bin/env python3
"""Simple CLI to run data build, analysis, and forecasting in sequence.

Runs the backend scripts as separate processes to avoid altering their internals.
It will attempt to use `ALL_CLEAN_DATA.csv` as the Excel input to `build_dataframe.py`
to avoid interactive upload. `dss_forecast.py` will be fed default inputs to run
all models with a default forecast horizon unless overridden.
"""
import sys
import subprocess
from pathlib import Path
import py_compile
import argparse


ROOT = Path(__file__).parent.resolve()
BACKEND = ROOT / 'backend'


def compile_py(path: Path):
    print(f"Compiling {path}...")
    try:
        py_compile.compile(str(path), doraise=True)
        print("  ✓ OK")
        return True
    except py_compile.PyCompileError as e:
        print(f"  ✗ Compile error: {e}")
        return False


def run_script(script: Path, args=None, input_text=None, env=None):
    cmd = [sys.executable, str(script)]
    if args:
        cmd += args
    print(f"\nRunning: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, input=input_text, text=True, capture_output=True, env=env)
        # Safely print stdout/stderr handling characters not mappable to the console encoding
        def safe_print(s):
            if not s:
                return
            try:
                print(s)
            except UnicodeEncodeError:
                enc = getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8'
                safe = s.encode('utf-8', errors='replace').decode(enc, errors='replace')
                print(safe)

        safe_print(res.stdout)
        if res.returncode != 0:
            print(f"Script exited with code {res.returncode}")
            safe_print(res.stderr)
        else:
            print("  ✓ Completed successfully")
        return res.returncode
    except Exception as exc:
        print(f"Failed to run script: {exc}")
        return 2


def main():
    parser = argparse.ArgumentParser(description="Run the data build, analysis, and forecasting sequence.")
    parser.add_argument('--dry-run', action='store_true', help='Print planned commands without executing')
    args = parser.parse_args()

    dry_run = args.dry_run

    # Targets
    build_script = BACKEND / 'build_dataframe.py'
    analyst_script = BACKEND / 'dss_analyst.py'
    forecast_script = BACKEND / 'dss_forecast.py'

    scripts = [build_script, analyst_script, forecast_script]

    # Quick compile check
    all_good = True
    for s in scripts:
        if not s.exists():
            print(f"Required script not found: {s}")
            all_good = False
            continue
        ok = compile_py(s)
        all_good = all_good and ok

    if not all_good:
        print("One or more scripts failed to compile. Aborting.")
        sys.exit(1)

    # 1) Build dataframe — use ALL_CLEAN_DATA.csv if present to avoid interactive upload
    excel_arg = None
    all_clean = ROOT / 'ALL_CLEAN_DATA.csv'
    if all_clean.exists():
        excel_arg = str(all_clean)
        print(f"Using {excel_arg} as input for build_dataframe.py to avoid interactive upload")

    build_args = ['--excel', excel_arg] if excel_arg else None
    if dry_run:
        print(f"DRY RUN: Would run: {sys.executable} {build_script} {' '.join(build_args) if build_args else ''}")
        code = 0
    else:
        code = run_script(build_script, args=build_args)
    if code != 0:
        print("build_dataframe.py failed — stopping sequence.")
        sys.exit(code)

    # 2) Run analyst
    if dry_run:
        print(f"DRY RUN: Would run: {sys.executable} {analyst_script}")
        code = 0
    else:
        code = run_script(analyst_script)
    if code != 0:
        print("dss_analyst.py failed — stopping sequence.")
        sys.exit(code)

    # 3) Run forecast non-interactively: select all (7) and 128 steps
    # Provide input: select '7' then newline then '128' then newline
    forecast_input = '7\n128\n'
    if dry_run:
        print(f"DRY RUN: Would run: {sys.executable} {forecast_script} (input: '7 then 128')")
        code = 0
    else:
        code = run_script(forecast_script, input_text=forecast_input)
    if code != 0:
        print("dss_forecast.py failed — sequence ended with errors.")
        sys.exit(code)

    print('\nAll steps completed successfully.')


if __name__ == '__main__':
    main()
