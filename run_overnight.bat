@echo off
REM ============================================================
REM  Alpha0 — Full Overnight Pipeline
REM  Run this from the project root:  run_overnight.bat
REM
REM  Estimated runtime on CPU:
REM    Step 1 (data):      30–60 min  (yfinance rate limits)
REM    Step 2 (train):     3–6 hours  (200k steps × 100 assets)
REM    Step 3 (evaluate):  5 min
REM    TOTAL:              ~4–7 hours
REM ============================================================

setlocal

REM ── CONFIGURE THESE ──────────────────────────────────────────
REM  Your FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
REM  If you don't have one, change --skip-fred below.
set FRED_API_KEY=34577da65b619f4acbf59b69f4d45741

REM  Training steps (50k is enough for a first overnight CPU run; use 200k+ with GPU)
set N_STEPS=50000

REM  Device: cpu  or  cuda  (if you have a GPU)
set DEVICE=cpu
REM ─────────────────────────────────────────────────────────────

echo.
echo ============================================================
echo  Alpha0 Overnight Pipeline  %date% %time%
echo ============================================================
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM ── Step 1: Fetch and build feature tensors ──────────────────
echo [1/4] Fetching data and building feature tensors...
echo       (This calls yfinance + FRED — takes 30-60 min)
echo.

if "%FRED_API_KEY%"=="YOUR_FRED_API_KEY_HERE" (
    echo       WARNING: FRED_API_KEY not set — skipping macro data.
    echo       Macro features (VIX, rates) will be zero-filled.
    echo       Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
    echo.
    python scripts\fetch_data.py --skip-fred
) else (
    python scripts\fetch_data.py
)

if errorlevel 1 (
    echo.
    echo ERROR: Data fetch failed. Check logs above.
    pause
    exit /b 1
)

echo.
echo [1/4] Data fetch complete.
echo.

REM ── Step 2: Train single SAC agent ───────────────────────────
echo [2/4] Training SAC agent (%N_STEPS% steps on %DEVICE%)...
echo       Checkpoints saved to: artifacts\models\latest\
echo.

python scripts\train.py ^
    --n-steps %N_STEPS% ^
    --device %DEVICE% ^
    --save-dir artifacts\models\latest

if errorlevel 1 (
    echo.
    echo ERROR: Training failed. Check logs above.
    pause
    exit /b 1
)

echo.
echo [2/4] Training complete.
echo.

REM ── Step 3: Walk-forward validation ──────────────────────────
echo [3/4] Running walk-forward validation (all 7 folds)...
echo       This retrains a model per fold — may take several hours.
echo       Skip this step by commenting out lines below if short on time.
echo.

python scripts\train.py ^
    --walk-forward ^
    --n-steps %N_STEPS% ^
    --device %DEVICE%

if errorlevel 1 (
    echo       Walk-forward failed or was skipped — continuing.
)

echo.
echo [3/4] Walk-forward complete (or skipped).
echo.

REM ── Step 4: Evaluate and generate report ─────────────────────
echo [4/4] Evaluating on test split and generating report...
echo.

python scripts\evaluate.py ^
    --checkpoint artifacts\models\latest\best.pt ^
    --split test ^
    --n-episodes 20 ^
    --benchmarks ^
    --walk-forward-csv artifacts\walk_forward\summary.csv ^
    --report-dir artifacts\reports\overnight_%date:~-4,4%%date:~-10,2%%date:~-7,2%

if errorlevel 1 (
    echo.
    echo ERROR: Evaluation failed. Check logs above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Pipeline complete!  %date% %time%
echo.
echo  Results:
echo    Model checkpoint:  artifacts\models\latest\best.pt
echo    Walk-forward CSV:  artifacts\walk_forward\summary.csv
echo    Dashboard report:  artifacts\reports\overnight_*\
echo ============================================================
echo.

pause
