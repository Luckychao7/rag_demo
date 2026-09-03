@echo off
cd /d "%~dp0"
echo ========================================
echo   Document QA Assistant - Launcher
echo ========================================
echo.

REM Check Ollama running
echo [1/3] Checking Ollama service...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Ollama not running, start it first: ollama serve
    echo [!] Then in another terminal run: ollama pull qwen2:7b-instruct-q4_K_M
    pause
    exit /b 1
)
echo      Ollama connected [OK]

REM Check model exists
echo [2/3] Checking model...
curl -s http://localhost:11434/api/tags | findstr "qwen2:7b" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Model qwen2:7b-instruct-q4_K_M not found
    echo     Downloading, please wait...
    ollama pull qwen2:7b-instruct-q4_K_M
)
echo      Model ready [OK]

REM Activate conda env (streamlit is only installed in doc_qa; without this, 'streamlit' is not found)
call C:\dev\Miniconda\condabin\activate.bat doc_qa
if errorlevel 1 (
    echo [!] Failed to activate doc_qa env, check conda at C:\dev\Miniconda
    pause
    exit /b 1
)
echo      doc_qa env activated [OK]

REM Start Streamlit
echo [3/3] Starting web UI...
echo.
echo ========================================
echo   Browser will open http://localhost:8501
echo   Press Ctrl+C to stop
echo ========================================
echo.
streamlit run app.py
