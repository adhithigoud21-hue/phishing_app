@echo off
echo ================================================================================
echo    ADVANCED PHISHING DETECTION SYSTEM - INSTALLATION SCRIPT
echo ================================================================================
echo.

echo [1/4] Activating Virtual Environment...
call .venv\Scripts\activate
if errorlevel 1 (
    echo ERROR: Could not activate virtual environment
    echo Please create one first with: python -m venv .venv
    pause
    exit /b 1
)
echo OK - Virtual environment activated
echo.

echo [2/4] Installing Dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo OK - All dependencies installed
echo.

echo [3/4] Training Model (this may take 2-5 minutes)...
python train_model_advanced.py
if errorlevel 1 (
    echo ERROR: Failed to train model
    pause
    exit /b 1
)
echo OK - Model trained successfully
echo.

echo [4/4] All done!
echo.
echo ================================================================================
echo    INSTALLATION COMPLETE!
echo ================================================================================
echo.
echo To run the application, execute:
echo    streamlit run streamlit_app.py
echo.
echo Or simply run: run_app.bat
echo.
pause
