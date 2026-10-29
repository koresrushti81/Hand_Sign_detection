@echo off
REM Hand Sign Recognition System Launcher
REM Windows Batch Script

echo.
echo ============================================
echo  Hand Sign Recognition System
echo ============================================
echo.
echo Choose an option:
echo.
echo 1. Run GUI Application (Recommended)
echo 2. Run Recognition Test
echo 3. Run Data Collection Tool
echo 4. Install Dependencies
echo 5. Exit
echo.

set /p choice="Select option (1-5): "

if "%choice%"=="1" (
    echo Starting GUI Application...
    python app.py
) else if "%choice%"=="2" (
    echo Starting Recognition Test...
    python test_improved.py
) else if "%choice%"=="3" (
    echo Starting Data Collection...
    python datacollection_improved.py
) else if "%choice%"=="4" (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo Done!
    pause
) else if "%choice%"=="5" (
    exit
) else (
    echo Invalid option
    pause
    goto start
)
