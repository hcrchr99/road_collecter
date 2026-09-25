@echo off
rem RoadCheck M-L2 v3.1 experiment: prompt v3 + 6 frames + CLAHE preprocessing
rem New labels (tang revised, schema v1.2 with biaoxian class) are already in place.
rem Resume-safe. Logs: out\llm_pipeline\run_*.log

set "ZHIPU_API_KEY="
for /f "tokens=2,*" %%a in ('reg query HKCU\Environment /v ZHIPU_API_KEY 2^>nul') do set "ZHIPU_API_KEY=%%b"
if "%ZHIPU_API_KEY%"=="" (
  echo [ERROR] ZHIPU_API_KEY not found in user environment variables
  pause
  exit /b 1
)

cd /d C:\Users\Admin\WorkBuddy\2026-09-18-23-02-29\_repo
set "PYTHON=C:\Users\Admin\.workbuddy\binaries\python\envs\default\Scripts\python.exe"

start "ML2-v31 cloud glm-4.6v 6f+clahe" cmd /k %PYTHON% -X utf8 -m llm_pipeline.run_all_ml2 --tag v31 --prompt prompts/judge_system_v3.md --frames 6 --preprocess clahe glm-4.6v

echo.
echo Cloud run window launched (keep it open):
echo   glm-4.6v, prompt v3, 6 frames, CLAHE   (concurrency 4, ~30-60 min)
echo.
echo NOTE: local qwen3-vl-8b runs are paused (too slow) since 2026-09-25.
echo Console logs also saved to _repo\out\llm_pipeline\run_*.log
echo Interrupted? Just double-click this bat again to resume.
pause
