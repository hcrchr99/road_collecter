@echo off
rem RoadCheck M-L2 v4.1 experiment: v4 recall-first + confidence semantics rebuilt
rem confidence = visual-evidence strength (decoupled from the defect decision),
rem so a post-hoc confidence threshold becomes a usable operating-point dial.
rem Resume-safe. Logs: out\llm_pipeline\run_*.log

set "ZHIPU_API_KEY="
for /f "tokens=2,*" %%a in ('reg query HKCU\Environment /v ZHIPU_API_KEY 2^>nul') do set "ZHIPU_API_KEY=%%b"
if "%ZHIPU_API_KEY%"=="" (
  echo [ERROR] ZHIPU_API_KEY not found in user environment variables
  pause
  exit /b 1
)
set "DEEPSEEK_API_KEY="
for /f "tokens=2,*" %%a in ('reg query HKCU\Environment /v DEEPSEEK_API_KEY 2^>nul') do set "DEEPSEEK_API_KEY=%%b"
if "%DEEPSEEK_API_KEY%"=="" (
  echo [ERROR] DEEPSEEK_API_KEY not found in user environment variables
  pause
  exit /b 1
)

cd /d C:\Users\Admin\WorkBuddy\2026-09-18-23-02-29\_repo
set "PYTHON=C:\Users\Admin\.workbuddy\binaries\python\envs\default\Scripts\python.exe"

start "ML2-v41 cloud glm-4.6v" cmd /k %PYTHON% -X utf8 -m llm_pipeline.run_all_ml2 --tag v41 --prompt prompts/judge_system_v41.md glm-4.6v
start "ML2-v41 cloud deepseek-v4.1f" cmd /k %PYTHON% -X utf8 -m llm_pipeline.run_all_ml2 --tag v41 --prompt prompts/judge_system_v41.md deepseek-v4.1f

echo.
echo Two cloud run windows launched (keep them open):
echo   glm-4.6v        prompt v4.1 (recall-first + evidence-based confidence)
echo   deepseek-v4.1f  same prompt (cross-vendor check)   (~30 min each)
echo.
echo Success criteria:
echo   1. confidence distribution non-degenerate (some defect calls at high/mid)
echo   2. high/mid defect precision clearly above low (threshold knob works)
echo   3. recall stays near v4 level (>=60%%)
echo Interrupted? Just double-click this bat again to resume.
pause
