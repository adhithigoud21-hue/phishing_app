# Run Streamlit App (PowerShell)
# Run this with: .\run_app.ps1

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "   STARTING PHISHING DETECTION SYSTEM" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "Starting Streamlit app..." -ForegroundColor Green
Write-Host "The app will open in your default browser at http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

streamlit run streamlit_app.py
