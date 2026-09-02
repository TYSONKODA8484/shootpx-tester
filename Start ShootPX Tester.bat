@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title ShootPX Tester Launcher

where streamlit >nul 2>nul
if errorlevel 1 (
    echo.
    echo   Streamlit isn't installed for this Python yet.
    echo   Open a terminal in this folder and run:  pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:menu
cls
echo.
echo   ShootPX Tester
echo   ================
echo.
echo   1. On-Model Shots
echo   2. Catalog Photoshoot
echo   3. Creative Photoshoot
echo   4. Recolor
echo   5. Exit
echo.
set "choice="
set /p choice="  Choose a number: "

if "%choice%"=="1" set "app=streamlit_app.py" & goto :run
if "%choice%"=="2" set "app=catalog_streamlit_app.py" & goto :run
if "%choice%"=="3" set "app=creative_streamlit_app.py" & goto :run
if "%choice%"=="4" set "app=recolor_streamlit_app.py" & goto :run
if "%choice%"=="5" goto :end

echo.
echo   Not a valid choice - try again.
pause >nul
goto :menu

:run
cls
echo.
echo   Starting %app% ...
echo   Your browser will open in a few seconds.
echo   To stop this tool, just close this window.
echo.
streamlit run "%app%"
echo.
echo   %app% has stopped.
echo.
pause
goto :menu

:end
exit
