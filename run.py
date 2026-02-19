#!/usr/bin/env python3
"""Operator 1 -- Interactive Terminal Launcher.

A user-friendly interface that guides you through running the
financial analysis pipeline step by step.

Usage:
    python run.py
"""

from __future__ import annotations

import os
import sys
import time
import socket
import shutil
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _color(text: str, code: str) -> str:
    """Wrap text in ANSI color codes (no-op on Windows without colorama)."""
    if os.name == "nt":
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(t: str) -> str:
    return _color(t, "32")


def _yellow(t: str) -> str:
    return _color(t, "33")


def _red(t: str) -> str:
    return _color(t, "31")


def _cyan(t: str) -> str:
    return _color(t, "36")


def _bold(t: str) -> str:
    return _color(t, "1")


def _dim(t: str) -> str:
    return _color(t, "2")


def _banner():
    print("")
    print(_bold(_cyan("  ================================================================")))
    print(_bold(_cyan("     OPERATOR 1 -- Financial Analysis Pipeline")))
    print(_bold(_cyan("  ================================================================")))
    print(_dim("  Bloomberg-style equity research powered by 20+ math models"))
    print(_dim("  Survival mode analysis | Ethical filters | Multi-horizon forecasts"))
    print("")


def _separator():
    print(_dim("  " + "-" * 60))


def _step(num: int, title: str):
    print("")
    print(_bold(f"  [{num}] {title}"))
    print("")


def _ok(msg: str):
    print(f"  {_green('[OK]')} {msg}")


def _warn(msg: str):
    print(f"  {_yellow('[!]')} {msg}")


def _err(msg: str):
    print(f"  {_red('[ERROR]')} {msg}")


def _info(msg: str):
    print(f"  {_dim('[i]')} {msg}")


def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"  > {msg}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("")
        sys.exit(0)
    return value if value else default


def _yes_no(msg: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        value = input(f"  > {msg} {suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("")
        sys.exit(0)
    if not value:
        return default
    return value in ("y", "yes")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_python_version() -> bool:
    v = sys.version_info
    if v.major >= 3 and v.minor >= 10:
        _ok(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    elif v.major >= 3 and v.minor >= 8:
        _warn(f"Python {v.major}.{v.minor}.{v.micro} (3.10+ recommended)")
        return True
    else:
        _err(f"Python {v.major}.{v.minor}.{v.micro} -- need 3.8+")
        return False


def check_internet() -> bool:
    """Check internet connectivity by attempting to reach a known host."""
    hosts = [
        ("api.worldbank.org", 443),
        ("financialmodelingprep.com", 443),
        ("1.1.1.1", 53),
    ]
    for host, port in hosts:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
            _ok(f"Internet connection active (reached {host})")
            return True
        except (socket.timeout, socket.error, OSError):
            continue
    _err("No internet connection detected")
    _info("This pipeline requires internet to fetch financial data.")
    return False


def check_dependencies() -> tuple[bool, list[str]]:
    """Check which key Python packages are installed."""
    required = {
        "requests": "HTTP requests",
        "pandas": "Data processing",
        "numpy": "Numerical computing",
        "yaml": "Config loading (pyyaml)",
    }
    optional = {
        "statsmodels": "Kalman filter, VAR models",
        "arch": "GARCH volatility models",
        "sklearn": "Tree ensembles, imputation",
        "torch": "LSTM deep learning",
        "hmmlearn": "Hidden Markov Models",
        "ruptures": "Structural break detection",
        "xgboost": "XGBoost tree ensemble",
        "matplotlib": "Chart generation",
        "dotenv": "python-dotenv (.env loading)",
    }

    missing_required: list[str] = []
    missing_optional: list[str] = []

    for pkg, desc in required.items():
        try:
            __import__(pkg)
            _ok(f"{pkg} -- {desc}")
        except ImportError:
            _err(f"{pkg} -- {desc} [MISSING]")
            missing_required.append(pkg)

    for pkg, desc in optional.items():
        try:
            __import__(pkg)
            _ok(f"{pkg} -- {desc}")
        except ImportError:
            _warn(f"{pkg} -- {desc} [not installed, some features limited]")
            missing_optional.append(pkg)

    return len(missing_required) == 0, missing_required


def check_api_keys() -> dict[str, str]:
    """Load or prompt for API keys."""
    env_path = Path(__file__).resolve().parent / ".env"

    # Try loading existing .env
    keys: dict[str, str] = {}
    if env_path.exists():
        _info(f"Found .env file at {env_path}")
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and v and not v.startswith("your_"):
                    keys[k] = v

    # Also check environment variables
    all_key_names = [
        "EULERPOOL_API_KEY", "EOD_API_KEY",
        "FMP_API_KEY", "GEMINI_API_KEY",
    ]
    for key_name in all_key_names:
        if key_name not in keys:
            env_val = os.environ.get(key_name)
            if env_val:
                keys[key_name] = env_val

    # Report status and prompt for missing keys
    # Equity provider: either Eulerpool or EOD must be set
    equity_keys = {
        "EULERPOOL_API_KEY": "Eulerpool (company financials)",
        "EOD_API_KEY": "EOD Historical Data (alternative to Eulerpool)",
    }
    always_required_keys = {
        "FMP_API_KEY": "FMP (OHLCV market data)",
        "GEMINI_API_KEY": "Google Gemini (AI reports)",
    }

    # Show equity provider status
    has_equity = any(k in keys for k in equity_keys)
    print(_dim("  Equity data provider (set one):"))
    for key_name, desc in equity_keys.items():
        if key_name in keys:
            masked = keys[key_name][:4] + "..." + keys[key_name][-4:]
            _ok(f"{key_name}: {masked} ({desc})")
        else:
            _warn(f"{key_name}: not set ({desc})")
    print("")

    # Show other required keys
    all_present = has_equity
    for key_name, desc in always_required_keys.items():
        if key_name in keys:
            masked = keys[key_name][:4] + "..." + keys[key_name][-4:]
            _ok(f"{key_name}: {masked} ({desc})")
        else:
            _warn(f"{key_name}: not set ({desc})")
            all_present = False

    if not all_present or not has_equity:
        print("")
        _info("Some API keys are missing. Enter them now or press Enter to skip.")
        _info("Keys will be saved to .env for future runs.")
        print("")

        # Prompt for equity provider if neither is set
        if not has_equity:
            print(_dim("  You need at least one equity data provider key:"))
            print(_dim("    1. Eulerpool (EULERPOOL_API_KEY)"))
            print(_dim("    2. EOD Historical Data (EOD_API_KEY) -- https://eodhd.com"))
            print("")
            value = _prompt("Enter EULERPOOL_API_KEY (or press Enter to use EOD instead)")
            if value:
                keys["EULERPOOL_API_KEY"] = value
            else:
                value = _prompt("Enter EOD_API_KEY (https://eodhd.com)")
                if value:
                    keys["EOD_API_KEY"] = value

        for key_name, desc in always_required_keys.items():
            if key_name not in keys:
                value = _prompt(f"Enter {key_name} ({desc})")
                if value:
                    keys[key_name] = value

        # Save to .env
        saveable = {**equity_keys, **always_required_keys}
        if any(k in keys for k in saveable):
            try:
                with open(env_path, "w") as f:
                    f.write("# Operator 1 -- API Keys\n")
                    f.write("# Auto-generated by run.py\n\n")
                    f.write("# Equity data provider (set one)\n")
                    for k, v in keys.items():
                        f.write(f"{k}={v}\n")
                _ok(f"API keys saved to {env_path}")
            except Exception as exc:
                _warn(f"Could not save .env: {exc}")

    # Set in environment for the pipeline
    for k, v in keys.items():
        os.environ[k] = v

    # Validate: at least one equity key + all always-required keys
    has_equity = any(k in keys for k in equity_keys)
    missing_required = [k for k in always_required_keys if k not in keys]
    if not has_equity:
        _err("No equity data provider key set (need EULERPOOL_API_KEY or EOD_API_KEY)")
        _info("The pipeline will fail without an equity data provider.")
    if missing_required:
        _err(f"Missing required keys: {', '.join(missing_required)}")
        _info("The pipeline will fail without these keys.")

    return keys


def estimate_runtime(skip_linked: bool, skip_models: bool) -> str:
    """Rough time estimate based on options."""
    if skip_models and skip_linked:
        return "~2-5 minutes (cache building only)"
    elif skip_models:
        return "~5-10 minutes (cache + linked entities, no models)"
    elif skip_linked:
        return "~15-30 minutes (models without linked entities)"
    else:
        return "~30-60 minutes (full analysis with all models)"


# ---------------------------------------------------------------------------
# Main interactive flow
# ---------------------------------------------------------------------------

def main() -> int:
    _clear()
    _banner()

    # ------------------------------------------------------------------
    # Step 1: System checks
    # ------------------------------------------------------------------
    _step(1, "System Checks")

    if not check_python_version():
        return 1

    _separator()

    if not check_internet():
        if not _yes_no("Continue without internet? (pipeline will likely fail)"):
            return 1

    # ------------------------------------------------------------------
    # Step 2: Dependencies
    # ------------------------------------------------------------------
    _step(2, "Checking Dependencies")

    deps_ok, missing = check_dependencies()

    if not deps_ok:
        print("")
        _err(f"Missing required packages: {', '.join(missing)}")
        if _yes_no("Install missing packages with pip?"):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
                check=False,
            )
            print("")
            _info("Re-checking dependencies...")
            deps_ok, missing = check_dependencies()
            if not deps_ok:
                _err("Still missing packages. Install manually: pip install -r requirements.txt")
                return 1
        else:
            _info("Run: pip install -r requirements.txt")
            return 1

    # ------------------------------------------------------------------
    # Step 3: API keys
    # ------------------------------------------------------------------
    _step(3, "API Keys")

    keys = check_api_keys()

    has_equity_key = ("EULERPOOL_API_KEY" in keys or "EOD_API_KEY" in keys)
    has_other_keys = all(k in keys for k in ["FMP_API_KEY", "GEMINI_API_KEY"])
    if not has_equity_key or not has_other_keys:
        _err("Cannot proceed without required API keys.")
        if not has_equity_key:
            _info("Need either EULERPOOL_API_KEY or EOD_API_KEY.")
        return 1

    # ------------------------------------------------------------------
    # Step 4: User inputs
    # ------------------------------------------------------------------
    _step(4, "Company Selection")

    print("  Enter the company identifiers:")
    print("")
    isin = _prompt("ISIN (e.g. US0378331005 for Apple)", "")
    if not isin:
        _err("ISIN is required.")
        return 1

    symbol = _prompt("FMP symbol (e.g. AAPL)", "")
    if not symbol:
        _err("FMP symbol is required.")
        return 1

    # ------------------------------------------------------------------
    # Step 5: Options
    # ------------------------------------------------------------------
    _step(5, "Pipeline Options")

    skip_linked = not _yes_no("Discover linked entities? (competitors, suppliers)", default=True)
    skip_models = not _yes_no("Run temporal models? (forecasting, burn-out)", default=True)
    gen_pdf = _yes_no("Generate PDF report? (requires pandoc)", default=False)

    # ------------------------------------------------------------------
    # Step 6: Confirmation
    # ------------------------------------------------------------------
    _step(6, "Confirmation")

    estimate = estimate_runtime(skip_linked, skip_models)

    print(f"  Target ISIN:      {_bold(isin)}")
    print(f"  FMP Symbol:       {_bold(symbol)}")
    print(f"  Linked entities:  {'Yes' if not skip_linked else 'Skip'}")
    print(f"  Temporal models:  {'Yes' if not skip_models else 'Skip'}")
    print(f"  PDF output:       {'Yes' if gen_pdf else 'No'}")
    print(f"  Estimated time:   {_yellow(estimate)}")
    print("")

    if not _yes_no("Start the pipeline?"):
        _info("Cancelled.")
        return 0

    # ------------------------------------------------------------------
    # Step 7: Run pipeline
    # ------------------------------------------------------------------
    _step(7, "Running Pipeline")

    start_time = time.time()

    cmd = [
        sys.executable, "main.py",
        "--isin", isin,
        "--symbol", symbol,
    ]
    if skip_linked:
        cmd.append("--skip-linked")
    if skip_models:
        cmd.append("--skip-models")
    if gen_pdf:
        cmd.append("--pdf")

    _info(f"Command: {' '.join(cmd)}")
    print("")
    _separator()
    print("")

    result = subprocess.run(cmd, check=False)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("")
    _separator()
    print("")

    if result.returncode == 0:
        _ok(f"Pipeline completed successfully in {minutes}m {seconds}s")
        print("")
        _info("Output files:")

        output_dir = Path("cache")
        if output_dir.exists():
            for f in sorted(output_dir.rglob("*")):
                if f.is_file():
                    size = f.stat().st_size
                    size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
                    print(f"    {f.relative_to('.')}  ({size_str})")

        print("")
        _info("Key files:")
        for key_file in [
            "cache/company_profile.json",
            "cache/report/analysis_report.md",
            "cache/report/analysis_report.pdf",
        ]:
            p = Path(key_file)
            if p.exists():
                print(f"    {_green('[exists]')} {key_file}")
            else:
                print(f"    {_dim('[  --  ]')} {key_file}")
    else:
        _err(f"Pipeline failed (exit code {result.returncode}) after {minutes}m {seconds}s")
        _info("Check the log output above for details.")
        return 1

    print("")
    _bold("  Done! Review the report in cache/report/analysis_report.md")
    print("")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("")
        print(_dim("  Interrupted."))
        sys.exit(130)
