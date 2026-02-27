@echo off
:: NASCAR Stats Weekly Updater
:: Runs every Monday at 12pm via Windows Task Scheduler
:: Pulls new race results, then recalculates fantasy scores.

set PROJECT=C:\Users\thoma\OneDrive\Desktop\Misc\Claude\Projects\NASCAR\.claude\worktrees\fervent-maxwell
set PYTHON=C:\Users\thoma\AppData\Local\Programs\Python\Python314\python.exe
set LOG=%PROJECT%\update_log.txt

echo. >> "%LOG%"
echo ======================================== >> "%LOG%"
echo Update started: %DATE% %TIME% >> "%LOG%"
echo ======================================== >> "%LOG%"

cd /d "%PROJECT%"

echo Fetching new race results... >> "%LOG%"
"%PYTHON%" fetch_races.py >> "%LOG%" 2>&1

echo Rebuilding fantasy scores... >> "%LOG%"
"%PYTHON%" build_fantasy.py >> "%LOG%" 2>&1

echo Update complete: %DATE% %TIME% >> "%LOG%"
