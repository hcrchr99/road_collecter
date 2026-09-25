@echo off
rem RoadCheck M-L2 experiment launcher (ASCII only - cmd parses bat in ANSI codepage)
rem Double-click to run. Two windows: cloud (3 GLM models serial) + local (qwen3-vl-8b, GPU).
rem Resume-safe: re-run this bat any time; already-judged events are skipped.
rem API key is read from user env var (registry), never stored in this file.

rem Read ZHIPU_API_KEY from HKCU\Environment (new cmd shells may predate setx)
set "ZHIPU_API_KEY="
for /f "tokens=2,*" %%a in ('reg query HKCU\Environment /v ZHIPU_API_KEY 2^>nul') do set "ZHIPU_API_KEY=%%b"
if "%ZHIPU_API_KEY%"=="" (
  echo [ERROR] ZHIPU_API_KEY not found in user environment variables
  pause
  exit /b 1
)

cd /d C:\Users\Admin\WorkBuddy\2026-09-18-23-02-29\_repo
set "PYTHON=C:\Users\Admin\.workbuddy\binaries\python\envs\default\Scripts\python.exe"

start "ML2-cloud glm-5.3-flash / 4.6v-flash / 4.6v" cmd /k %PYTHON% -X utf8 -m llm_pipeline.run_all_ml2 glm-5.3-flash glm-4.6v-flash glm-4.6v
start "ML2-local qwen3-vl-8b (GPU)" cmd /k %PYTHON% -X utf8 -m llm_pipeline.run_all_ml2 qwen3-vl-8b

echo.
echo Two run windows launched (keep them open):
echo   Window 1 cloud: glm-5.3-flash -^> glm-4.6v-flash -^> glm-4.6v   (concurrency 4, ~1-2 h)
echo   Window 2 local: qwen3-vl-8b-16k                                (GPU serial, ~2-4 h)
echo.
echo When all done, eval report: _repo\out\llm_pipeline\ml2_eval.md
echo Interrupted? Just double-click this bat again to resume.
pause
