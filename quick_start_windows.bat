@echo off
setlocal enabledelayedexpansion

rem YOLOv8 Object Detection Inference Engine - Quick Start (Windows)
rem This script builds and runs the basic tests automatically on Windows.

title YOLOv8 Inference - Quick Start (Windows)
echo ==============================================================
echo   YOLOv8 Object Detection Inference Engine - QUICK START
echo ==============================================================
echo.

rem Resolve project root to the directory of this script
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

rem Check for Microsoft C++ build tools (cl.exe)
where cl >nul 2>nul
if errorlevel 1 (
  echo [!] Microsoft C++ Build Tools not found.
  echo     Please install Visual Studio (Desktop development with C++)
  echo     or Build Tools for Visual Studio, then run this again.
  echo     Download: https://visualstudio.microsoft.com/downloads/
  pause
  exit /b 1
)

rem Check ONNX Runtime (Windows) folder
set ORT_DIR=%CD%\onnxruntime-windows-x64-1.17.0
if not exist "%ORT_DIR%" (
  echo [!] ONNX Runtime for Windows not found at:
  echo     %ORT_DIR%
  echo     Please download and extract: onnxruntime-win-x64-1.17.0 and rename to onnxruntime-windows-x64-1.17.0
  echo     Releases: https://github.com/microsoft/onnxruntime/releases
  pause
  exit /b 1
)

rem Ensure DLL path is available at runtime
set PATH=%ORT_DIR%\bin;%PATH%

echo [*] Building test_basic.exe
cl /nologo /std:c++17 /EHsc ^
  /I "%ORT_DIR%\include" ^
  test_basic.cpp ^
  "%ORT_DIR%\lib\onnxruntime.lib" ^
  /Fe:test_basic.exe >nul
if errorlevel 1 (
  echo [x] Build failed: test_basic.cpp
  pause
  exit /b 1
)

echo [*] Building simple_inference.exe
cl /nologo /std:c++17 /EHsc ^
  /I "%ORT_DIR%\include" ^
  simple_inference.cpp ^
  "%ORT_DIR%\lib\onnxruntime.lib" ^
  /Fe:simple_inference.exe >nul
if errorlevel 1 (
  echo [x] Build failed: simple_inference.cpp
  pause
  exit /b 1
)

echo.
echo [*] Running ONNX Runtime basic test ...
echo --------------------------------------------------------------
test_basic.exe
if errorlevel 1 (
  echo [x] test_basic failed.
  pause
  exit /b 1
)

echo.
echo [*] Running YOLOv8 simple inference ...
echo --------------------------------------------------------------
if not exist yolov8n.onnx (
  echo [!] yolov8n.onnx not found in project root.
  echo     Please place the model file here or run models\convert_model.py (Python).
  pause
  exit /b 1
)
simple_inference.exe yolov8n.onnx
if errorlevel 1 (
  echo [x] simple_inference failed.
  pause
  exit /b 1
)

echo.
echo [OK] Quick start completed successfully!
echo     - test_basic.exe ran successfully
echo     - simple_inference.exe ran successfully
echo.
echo Optional: Build the full application with OpenCV (see README)
pause
endlocal


