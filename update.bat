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
echo [1/5] Fetching new race results... >> "%LOG%"
"%PYTHON%" -u fetch_races.py >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: fetch_races.py failed with code %ERRORLEVEL%. Aborting. >> "%LOG%"
    echo Update FAILED: %DATE% %TIME% >> "%LOG%"
    exit /b 1
)
echo fetch_races OK >> "%LOG%"

echo [2/5] Rebuilding fantasy scores... >> "%LOG%"
"%PYTHON%" -u build_fantasy.py >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: build_fantasy.py failed with code %ERRORLEVEL%. Aborting. >> "%LOG%"
    echo Update FAILED: %DATE% %TIME% >> "%LOG%"
    exit /b 1
)
echo build_fantasy OK >> "%LOG%"

echo [3/5] Generating website... >> "%LOG%"
"%PYTHON%" -u report.py >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: report.py failed with code %ERRORLEVEL%. Aborting. >> "%LOG%"
    echo Update FAILED: %DATE% %TIME% >> "%LOG%"
    exit /b 1
)
echo report OK >> "%LOG%"

:: Commit the fresh index.html to the worktree branch first,
:: then merge that branch into main and push.
echo [4/5] Committing index.html... >> "%LOG%"
git add index.html >> "%LOG%" 2>&1
git commit -m "Auto-update: %DATE%" >> "%LOG%" 2>&1
:: exit code 1 from git commit means "nothing to commit" -- that's fine, don't abort
if errorlevel 2 (
    echo ERROR: git commit failed unexpectedly. Aborting. >> "%LOG%"
    echo Update FAILED: %DATE% %TIME% >> "%LOG%"
    exit /b 1
)
echo git commit OK >> "%LOG%"

echo [5/5] Merging to main and pushing... >> "%LOG%"
git -C "%MAINREPO%" merge --allow-unrelated-histories -m "Auto-update: %DATE%" claude/fervent-maxwell >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: git merge failed with code %ERRORLEVEL%. Aborting push. >> "%LOG%"
    echo Update FAILED: %DATE% %TIME% >> "%LOG%"
    exit /b 1
)
git -C "%MAINREPO%" push origin main >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: git push failed with code %ERRORLEVEL%. >> "%LOG%"
    echo Update FAILED: %DATE% %TIME% >> "%LOG%"
    exit /b 1
)
echo git push OK >> "%LOG%"

echo ======================================== >> "%LOG%"
echo Update COMPLETE: %DATE% %TIME% >> "%LOG%"
echo ======================================== >> "%LOG%"
