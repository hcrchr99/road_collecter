@echo off
rem RoadCheck M-L2 v3 experiment: prompt v3 (schema v1.2 + "biaoxian" class) + 6 frames
rem Runs glm-4.6v (cloud, concurrency 4) and local qwen3-vl-8b with 6 frames per event.
rem Resume-safe: re-run any time. Logs: out\llm_pipeline\run_*.log

set "ZHIPU_API_KEY="
for /f "tokens=2,*" %%a in ('reg query HKCU\Environment /v ZHIPU_API_KEY 2^>nul') do set "ZHIPU_API_KEY=%%b"
if "%ZHIPU_API_KEY%"=="" (
  echo [ERROR] ZHIPU_API_KEY not found in user environment variables
  pause
  exit /b 1
)

cd /d C:\Users\Admin\WorkBuddy\2026-09-18-23-02-29\_repo
set "PYTHON=C:\Users\Admin\.workbuddy\binaries\python\envs\default\Scripts\python.exe"

start "ML2-v3 cloud glm-4.6v 6frames" cmd /k %PYTHON% -X utf8 -m llm_pipeline.run_all_ml2 --tag v3 --prompt prompts/judge_system_v3.md --frames 6 glm-4.6v
start "ML2-v3 local qwen3-vl-8b 6frames" cmd /k %PYTHON% -X utf8 -m llm_pipeline.run_all_ml2 --tag v3 --prompt prompts/judge_system_v3.md --frames 6 qwen3-vl-8b

echo.
echo Two v3 run windows launched (keep them open):
echo   Window 1 cloud: glm-4.6v, prompt v3, 6 frames   (concurrency 4, ~30-60 min)
echo   Window 2 local: qwen3-vl-8b, prompt v3, 6 frames (GPU serial, slower)
echo.
echo v3 changes: 11-class vocab (+biaoxian/road-marking), 6 frames per event.
echo Console logs also saved to _repo\out\llm_pipeline\run_*.log
echo Eval: _repo\out\llm_pipeline\ml2_eval.md  (auto-generated after run)
echo Interrupted? Just double-click this bat again to resume.
pause
