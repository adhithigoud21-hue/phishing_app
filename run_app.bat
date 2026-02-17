@echo off
echo ================================================================================
echo    STARTING PHISHING DETECTION SYSTEM
echo ================================================================================
echo.

echo Activating virtual environment...
call .venv\Scripts\activate

echo.
echo Starting Streamlit app...
echo The app will open in your default browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run streamlit_app.py
