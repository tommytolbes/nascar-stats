@echo off
:: NASCAR Stats Weekly Updater
:: Runs every Monday at 12pm via Windows Task Scheduler
:: Pulls new race results, recalculates fantasy scores,
:: regenerates the website, and pushes to GitHub Pages.

set PROJECT=C:\Users\thoma\OneDrive\Desktop\Misc\Claude\Projects\NASCAR\.claude\worktrees\fervent-maxwell
set MAINREPO=C:\Users\thoma\OneDrive\Desktop\Misc\Claude\Projects\NASCAR
set PYTHON=C:\Users\thoma\AppData\Local\Programs\Python\Python314\python.exe
set LOG=%PROJECT%\update_log.txt

echo. >> "%LOG%"
echo ======================================== >> "%LOG%"
echo Update started: %DATE% %TIME% >> "%LOG%"
echo ======================================== >> "%LOG%"

cd /d "%PROJECT%"

:: -u = unbuffered output so every line is written to the log immediately
echo Fetching new race results... >> "%LOG%"
"%PYTHON%" -u fetch_races.py >> "%LOG%" 2>&1
echo fetch_races exit code: %ERRORLEVEL% >> "%LOG%"

echo Rebuilding fantasy scores... >> "%LOG%"
"%PYTHON%" -u build_fantasy.py >> "%LOG%" 2>&1
echo build_fantasy exit code: %ERRORLEVEL% >> "%LOG%"

echo Generating website... >> "%LOG%"
"%PYTHON%" -u report.py >> "%LOG%" 2>&1
echo report exit code: %ERRORLEVEL% >> "%LOG%"

:: Commit the fresh index.html to the worktree branch first,
:: then merge that branch into main and push.
echo Committing index.html... >> "%LOG%"
git add index.html >> "%LOG%" 2>&1
git commit -m "Auto-update: %DATE%" >> "%LOG%" 2>&1
echo git commit exit code: %ERRORLEVEL% >> "%LOG%"

echo Merging to main and pushing... >> "%LOG%"
git -C "%MAINREPO%" merge --allow-unrelated-histories -m "Auto-update: %DATE%" claude/fervent-maxwell >> "%LOG%" 2>&1
git -C "%MAINREPO%" push origin main >> "%LOG%" 2>&1
echo git push exit code: %ERRORLEVEL% >> "%LOG%"

echo Update complete: %DATE% %TIME% >> "%LOG%"
