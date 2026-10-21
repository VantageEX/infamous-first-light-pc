@echo off
rem Starts the game under RenderDoc (portable build in tools\renderdoc): F12 in the game window
rem writes a capture to captures\. The Vulkan loader ignores layer paths from the environment in
rem elevated processes, so the layer is registered once with
rem   renderdoccmd vulkanlayer --register --system   (undo: --unregister)
rem and stays off unless ENABLE_VULKAN_RENDERDOC_CAPTURE=1, which only this script sets.
setlocal
set "ROOT=%~dp0.."
set "RDOC=%ROOT%\tools\renderdoc\RenderDoc_1.46_64"
if not exist "%RDOC%\renderdoccmd.exe" (
  echo RenderDoc not found in %RDOC%
  exit /b 1
)
if not exist "%ROOT%\captures" mkdir "%ROOT%\captures"

set "ENABLE_VULKAN_RENDERDOC_CAPTURE=1"

if "%FL_GAME_DIR%"=="" set "FL_GAME_DIR=%ROOT%\CUSA00897"
set "TLG_GAME_DIR=%FL_GAME_DIR%"
set "TLG_DATA_DIR=%ROOT%\data"
set "TLG_USER_DIR=%ROOT%\data\user"
if "%TLG_LANGUAGE%"=="" set "TLG_LANGUAGE=8"
if "%TLG_OCCLUSION_MODE%"=="" set "TLG_OCCLUSION_MODE=4"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

"%RDOC%\renderdoccmd.exe" capture --opt-hook-children -w -d "%ROOT%\port" -c "%ROOT%\captures\firstlight" "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\port\run.py" > "%ROOT%\data\logs\renderdoc-run.log" 2>&1
echo Exit code %ERRORLEVEL%. Captures:
dir /b "%ROOT%\captures"
