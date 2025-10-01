@echo off
setlocal enabledelayedexpansion

rem Build the full application on Windows (requires OpenCV for Windows)
title YOLOv8 Inference - Build Full (Windows)

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

where cl >nul 2>nul
if errorlevel 1 (
  echo [!] Microsoft C++ Build Tools not found.
  echo     Open the "x64 Native Tools Command Prompt for VS" and run again.
  pause
  exit /b 1
)

set ORT_DIR=%CD%\onnxruntime-windows-x64-1.17.0
if not exist "%ORT_DIR%" (
  echo [!] ONNX Runtime not found at %ORT_DIR%
  pause
  exit /b 1
)

if "%OPENCV_DIR%"=="" (
  echo [!] OPENCV_DIR not set. Example: set OPENCV_DIR=C:\opencv\build
  pause
  exit /b 1
)

set PATH=%ORT_DIR%\bin;%OPENCV_DIR%\x64\vc16\bin;%PATH%

echo [*] Building inference_engine.exe
cl /nologo /std:c++17 /EHsc ^
  /I headers ^
  /I "%ORT_DIR%\include" ^
  /I "%OPENCV_DIR%\include" ^
  src\main.cpp src\infer_engine.cpp src\preprocess.cpp src\nms.cpp src\frame_queue.cpp src\frame.cpp ^
  "%ORT_DIR%\lib\onnxruntime.lib" ^
  "%OPENCV_DIR%\x64\vc16\lib\opencv_world4*.lib" ^
  /Fe:inference_engine.exe

if errorlevel 1 (
  echo [x] Build failed.
  pause
  exit /b 1
)

echo [OK] Build completed: inference_engine.exe
echo Run examples:
echo   inference_engine.exe --model yolov8n.onnx --video 0
echo   inference_engine.exe --model yolov8n.onnx --video data\sample_video.mp4 --conf 0.3
pause
endlocal


