# Advanced Phishing Detection System - Installation Script (PowerShell)
# Run this with: .\install.ps1

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "   ADVANCED PHISHING DETECTION SYSTEM - INSTALLATION SCRIPT" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if virtual environment exists
Write-Host "[1/4] Checking Virtual Environment..." -ForegroundColor Yellow
if (Test-Path ".venv\Scripts\activate.ps1") {
    Write-Host "OK - Virtual environment found" -ForegroundColor Green
    & .venv\Scripts\Activate.ps1
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK - Virtual environment created" -ForegroundColor Green
        & .venv\Scripts\Activate.ps1
    } else {
        Write-Host "ERROR: Could not create virtual environment" -ForegroundColor Red
        Write-Host "Please ensure Python is installed and in your PATH" -ForegroundColor Red
        pause
        exit 1
    }
}
Write-Host ""

# Step 2: Upgrade pip
Write-Host "[2/5] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host ""

# Step 3: Install Dependencies
Write-Host "[3/5] Installing Dependencies (this may take a few minutes)..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK - All dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Write-Host "Please check your internet connection and try again" -ForegroundColor Red
    pause
    exit 1
}
Write-Host ""

# Step 4: Verify installation
Write-Host "[4/5] Verifying installation..." -ForegroundColor Yellow
python -c "import pandas, sklearn, streamlit, plotly, xgboost; print('All packages imported successfully!')"
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK - All packages verified" -ForegroundColor Green
} else {
    Write-Host "WARNING: Some packages may not be properly installed" -ForegroundColor Yellow
}
Write-Host ""

# Step 5: Train Model
Write-Host "[5/5] Training Model (this may take 2-5 minutes)..." -ForegroundColor Yellow
Write-Host "Please wait while the model trains on 30,000+ samples..." -ForegroundColor Cyan
python train_model_advanced.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK - Model trained successfully!" -ForegroundColor Green
} else {
    Write-Host "ERROR: Failed to train model" -ForegroundColor Red
    Write-Host "Please check the error messages above" -ForegroundColor Red
    pause
    exit 1
}
Write-Host ""

Write-Host "================================================================================" -ForegroundColor Green
Write-Host "   INSTALLATION COMPLETE!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run the application, execute:" -ForegroundColor Cyan
Write-Host "   streamlit run streamlit_app.py" -ForegroundColor White
Write-Host ""
Write-Host "Or simply run:" -ForegroundColor Cyan
Write-Host "   .\run_app.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
