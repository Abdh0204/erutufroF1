# Operator 1 -- Financial Analysis Pipeline

A Bloomberg-style company analysis tool that builds a 2-year daily financial cache, runs 20+ mathematical models, and generates a comprehensive investment report.

---

## What does this app do?

You give it a company (like Apple or Tesla), and it:

1. Downloads 2 years of financial data (stock prices, balance sheets, cash flow)
2. Fetches country-level economic data (inflation, GDP, unemployment)
3. Discovers related companies (competitors, suppliers, customers)
4. Analyzes survival risk, debt health, and market stability
5. Runs AI models to predict future stock behavior
6. Generates a professional investment report (Markdown + optional PDF)

---

## Step-by-Step Setup on Linux Mint

### Step 1: Open a terminal

Click on the **Menu** (bottom-left corner) and search for **Terminal**, then click it.

You should see a black/dark window with a blinking cursor. This is where you will type commands.

### Step 2: Install Python (if you don't have it)

Linux Mint usually comes with Python pre-installed. Check by typing:

```bash
python3 --version
```

You should see something like `Python 3.10.12`. If you get an error, install it:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git -y
```

Type your password when asked (you won't see characters as you type -- that's normal).

### Step 3: Download the project from GitHub

```bash
cd ~/Desktop
git clone https://github.com/abdh0saifelden-png/Turtleiu.git
cd Turtleiu
```

This creates a folder called `Turtleiu` on your Desktop with all the code inside.

### Step 4: Create a virtual environment (keeps things clean)

```bash
python3 -m venv venv
source venv/bin/activate
```

Your terminal prompt should now start with `(venv)` -- this means you are inside the virtual environment.

**Important:** Every time you open a new terminal to use this app, run:
```bash
cd ~/Desktop/Turtleiu
source venv/bin/activate
```

### Step 5: Install the required packages

```bash
pip install -r requirements.txt
```

This downloads all the libraries the app needs. It may take a few minutes. You will see a lot of text scrolling -- that's normal.

If you see errors about `torch` (PyTorch), don't worry -- it's optional. The app will work without it (some advanced AI models just won't be available).

### Step 6: Get your API keys

The app needs API keys to access financial data. You need **three** keys:

| Key | What it's for | Where to get it |
|-----|--------------|-----------------|
| **EULERPOOL_API_KEY** | Company financial statements | Sign up at [eulerpool.com](https://www.eulerpool.com) |
| **FMP_API_KEY** | Stock price data (OHLCV) | Sign up at [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs/) |
| **GEMINI_API_KEY** | AI report generation | Sign up at [ai.google.dev](https://ai.google.dev/) |

Most of these have free tiers that are enough to get started.

### Step 7: Run the app

```bash
python3 run.py
```

The app will guide you through everything:

1. **System check** -- verifies Python, internet, and packages
2. **API keys** -- asks you to enter your keys (only the first time -- it saves them)
3. **Company selection** -- asks for the company ISIN and stock symbol
4. **Options** -- lets you choose what analysis to run
5. **Execution** -- runs the pipeline and shows progress
6. **Results** -- tells you where to find the output files

### What is an ISIN? What is a symbol?

- **ISIN** is like a company's passport number. Every publicly traded company has one.
  - Example: Apple's ISIN is `US0378331005`
  - Example: Tesla's ISIN is `US88160R1014`
  - You can find it by searching "[company name] ISIN" on Google

- **Symbol** (also called ticker) is the short code used on stock exchanges.
  - Example: Apple = `AAPL`
  - Example: Tesla = `TSLA`
  - Example: SAP = `SAP`

---

## Finding the Results

After the pipeline finishes, your results are in the `cache/` folder:

```
Turtleiu/
  cache/
    company_profile.json      <-- All the raw analysis data
    report/
      analysis_report.md      <-- The investment report (open with any text editor)
      analysis_report.pdf     <-- PDF version (if you chose PDF output)
      charts/                 <-- Chart images (PNG files)
```

To **read the report**, you can:

- Open `cache/report/analysis_report.md` with any text editor (right-click > Open With > Text Editor)
- For a nicer view, open it in a Markdown viewer (like VS Code, or paste it into [dillinger.io](https://dillinger.io))
- If you generated a PDF, open `analysis_report.pdf` with your PDF viewer

---

## Quick Reference (for repeat use)

After the first setup, running the app again is just:

```bash
cd ~/Desktop/Turtleiu
source venv/bin/activate
python3 run.py
```

Or if you prefer typing everything directly (no interactive prompts):

```bash
cd ~/Desktop/Turtleiu
source venv/bin/activate
python3 main.py --isin US0378331005 --symbol AAPL
```

### Useful command-line options

| Command | What it does |
|---------|-------------|
| `python3 run.py` | Interactive mode (guides you step by step) |
| `python3 main.py --isin ISIN --symbol SYM` | Run with specific company |
| `python3 main.py --isin ISIN --symbol SYM --skip-models` | Faster run (skip AI models) |
| `python3 main.py --isin ISIN --symbol SYM --skip-linked` | Skip finding related companies |
| `python3 main.py --isin ISIN --symbol SYM --pdf` | Also generate a PDF report |
| `python3 main.py --isin ISIN --symbol SYM --report-only` | Re-generate report from existing data |
| `python3 main.py --help` | Show all available options |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'xxx'"

You need to install the missing package:
```bash
pip install xxx
```

Or re-run the full install:
```bash
pip install -r requirements.txt
```

### "Missing required API keys"

Make sure you entered your API keys. You can re-enter them by editing the `.env` file:
```bash
nano .env
```

It should look like:
```
EULERPOOL_API_KEY=your_actual_key_here
FMP_API_KEY=your_actual_key_here
GEMINI_API_KEY=your_actual_key_here
```

Press `Ctrl+O` to save, then `Ctrl+X` to exit.

### "No internet connection detected"

Make sure you are connected to Wi-Fi or Ethernet. The app needs internet to download financial data.

### The pipeline takes too long

Use the faster options:
```bash
python3 main.py --isin US0378331005 --symbol AAPL --skip-linked --skip-models
```

This skips the heavy AI modeling and just builds the data cache + basic report. Takes about 2-5 minutes instead of 30-60 minutes.

### "pip: command not found"

Use `pip3` instead of `pip`:
```bash
pip3 install -r requirements.txt
```

Or install pip:
```bash
sudo apt install python3-pip -y
```

### PDF generation failed

PDF generation requires `pandoc`. Install it:
```bash
sudo apt install pandoc texlive-xetex -y
```

---

## Project Structure (for developers)

```
Turtleiu/
  run.py                  # Interactive launcher (start here)
  main.py                 # CLI entry point (for scripted use)
  .env.example            # API key template
  .env                    # Your actual API keys (not committed to git)
  requirements.txt        # Python dependencies
  config/                 # YAML configuration files
    survival_hierarchy.yml    # Tier weights and variable mappings
    global_config.yml         # Timeouts, retries, cache settings
    world_bank_indicator_map.yml  # Macro indicator codes
  operator1/              # Main Python package
    clients/              # API clients (Eulerpool, FMP, Gemini, World Bank)
    models/               # Forecasting, regime detection, Monte Carlo
    analysis/             # Survival mode, ethical filters, vanity analysis
    features/             # Feature engineering, derived variables
    estimation/           # Missing value estimation (Sudoku inference)
    report/               # Report generation, charts, PDF
    steps/                # Pipeline steps (verification, extraction, caching)
    quality/              # Data quality checks
  tests/                  # Test suite
  cache/                  # Output directory (created at runtime)
  notebooks/              # Kaggle notebook version
  plans/                  # Implementation plans and design docs
```

---

## License

This project is for educational and research purposes.
