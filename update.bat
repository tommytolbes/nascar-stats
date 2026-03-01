@echo off
:: NASCAR Stats Weekly Updater
:: Runs every Monday at 12pm via Windows Task Scheduler
:: Pulls new race results, recalculates fantasy scores,
:: regenerates the website, and pushes to GitHub Pages.

set PROJECT=C:\Users\thoma\OneDrive\Desktop\Misc\Claude\Projects\NASCAR\.claude\worktrees\fervent-maxwell
set MAINREPO=C:\Users\thoma\OneDrive\Desktop\Misc\Claude\Projects\NASCAR
set PYTHON=C:\Users\thoma\AppData\Local\Programs\Python\Python314\python.exe
set LOG=%PROJECT%\update_log.txt
set GIT=C:\Program Files\Git\bin\git.exe

echo. >> "%LOG%"
echo ======================================== >> "%LOG%"
echo Update started: %DATE% %TIME% >> "%LOG%"
echo ======================================== >> "%LOG%"

cd /d "%PROJECT%"

echo Fetching new race results... >> "%LOG%"
"%PYTHON%" fetch_races.py >> "%LOG%" 2>&1

echo Rebuilding fantasy scores... >> "%LOG%"
"%PYTHON%" build_fantasy.py >> "%LOG%" 2>&1

echo Generating website... >> "%LOG%"
"%PYTHON%" report.py >> "%LOG%" 2>&1

echo Pushing to GitHub... >> "%LOG%"
"%GIT%" -C "%MAINREPO%" merge --allow-unrelated-histories -m "Auto-update: %DATE%" claude/fervent-maxwell >> "%LOG%" 2>&1
"%GIT%" -C "%MAINREPO%" push origin main >> "%LOG%" 2>&1

echo Update complete: %DATE% %TIME% >> "%LOG%"
