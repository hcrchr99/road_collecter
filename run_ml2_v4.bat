@echo off
rem RoadCheck M-L2 v4 experiment: binary-first prompt (3 keyframes, v1-style config)
rem Hypothesis: focusing the prompt on defect/non-defect lowers false defects
rem and lifts the binary metric past the gate. Fine labels are secondary now.
rem Resume-safe. Logs: out\llm_pipeline\run_*.log

rem Read API keys from HKCU\Environment (new cmd shells may predate setx)
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

start "ML2-v4 cloud glm-4.6v binary-first" cmd /k %PYTHON% -X utf8 -m llm_pipeline.run_all_ml2 --tag v4 --prompt prompts/judge_system_v4.md glm-4.6v
start "ML2-v4 cloud deepseek-v4.1f binary-first" cmd /k %PYTHON% -X utf8 -m llm_pipeline.run_all_ml2 --tag v4 --prompt prompts/judge_system_v4.md deepseek-v4.1f

echo.
echo Two cloud run windows launched (keep them open):
echo   glm-4.6v        prompt v4 (binary-first), 3 keyframes   (concurrency 4, ~30 min)
echo   deepseek-v4.1f  prompt v4 (binary-first), 3 keyframes   (concurrency 4, cross-vendor)
echo.
echo Eval with binary + recall metrics: _repo\out\llm_pipeline\ml2_eval.md
echo Interrupted? Just double-click this bat again to resume.
pause
