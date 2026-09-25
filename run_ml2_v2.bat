@echo off
rem RoadCheck M-L2 v2 experiment launcher (prompt v2: anti-underreport of joints/rough)
rem Runs glm-4.6v and local qwen3-vl-8b with prompt v2, results in separate _v2 dirs.
rem Resume-safe: re-run any time. Logs also written to out\llm_pipeline\run_*.log

set "ZHIPU_API_KEY="
for /f "tokens=2,*" %%a in ('reg query HKCU\Environment /v ZHIPU_API_KEY 2^>nul') do set "ZHIPU_API_KEY=%%b"
if "%ZHIPU_API_KEY%"=="" (
  echo [ERROR] ZHIPU_API_KEY not found in user environment variables
  pause
  exit /b 1
)

cd /d C:\Users\Admin\WorkBuddy\2026-09-18-23-02-29\_repo
set "PYTHON=C:\Users\Admin\.workbuddy\binaries\python\envs\default\Scripts\python.exe"

start "ML2-v2 cloud glm-4.6v (prompt v2)" cmd /k %PYTHON% -X utf8 -m llm_pipeline.run_all_ml2 --tag v2 --prompt prompts/judge_system_v2.md glm-4.6v
start "ML2-v2 local qwen3-vl-8b (prompt v2)" cmd /k %PYTHON% -X utf8 -m llm_pipeline.run_all_ml2 --tag v2 --prompt prompts/judge_system_v2.md qwen3-vl-8b

echo.
echo Two v2 run windows launched (keep them open):
echo   Window 1 cloud: glm-4.6v prompt-v2        (concurrency 4, ~30-60 min)
echo   Window 2 local: qwen3-vl-8b prompt-v2     (GPU serial, ~2-4 h)
echo.
echo Console logs are ALSO saved to _repo\out\llm_pipeline\run_*.log
echo Eval report: _repo\out\llm_pipeline\ml2_eval.md
echo Interrupted or crashed? Just double-click this bat again to resume,
echo and check run_*.log for the error.
pause
