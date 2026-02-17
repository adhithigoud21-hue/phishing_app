# STEP-BY-STEP INSTALLATION GUIDE
# Run each command one by one

Write-Host @"
================================================================================
   MANUAL INSTALLATION GUIDE - RUN THESE COMMANDS ONE BY ONE
================================================================================

STEP 1: Activate Virtual Environment
-------------------------------------
"@ -ForegroundColor Cyan

Write-Host ".venv\Scripts\Activate.ps1" -ForegroundColor Yellow

Write-Host @"

STEP 2: Upgrade pip (optional but recommended)
-----------------------------------------------
"@ -ForegroundColor Cyan

Write-Host "python -m pip install --upgrade pip" -ForegroundColor Yellow

Write-Host @"

STEP 3: Install Dependencies
-----------------------------
"@ -ForegroundColor Cyan

Write-Host "pip install -r requirements.txt" -ForegroundColor Yellow

Write-Host @"

STEP 4: Train the Model (2-5 minutes)
--------------------------------------
"@ -ForegroundColor Cyan

Write-Host "python train_model_advanced.py" -ForegroundColor Yellow

Write-Host @"

STEP 5: Run the Streamlit App
------------------------------
"@ -ForegroundColor Cyan

Write-Host "streamlit run streamlit_app.py" -ForegroundColor Yellow

Write-Host @"

================================================================================
   QUICK COMMANDS (Copy and paste these)
================================================================================
"@ -ForegroundColor Green

Write-Host @"
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model_advanced.py
streamlit run streamlit_app.py
"@ -ForegroundColor White

Write-Host @"

================================================================================
   OR USE AUTOMATED INSTALLATION
================================================================================
"@ -ForegroundColor Green

Write-Host ".\install.ps1" -ForegroundColor Yellow

Write-Host ""
