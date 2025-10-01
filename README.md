# YOLOv8 Object Detection Inference Engine — Windows Run Guide

This is a minimal, step-by-step guide to build and run the project on Windows.

Quick flow:
- Double-click `quick_start_windows.bat` (builds and runs tests automatically).
- Optional: `build_windows_full.bat` for full app with OpenCV.

Prerequisites (no code changes needed):
- Install Visual Studio Build Tools or Visual Studio 2019/2022 with "Desktop development with C++".
- Download ONNX Runtime (Windows x64, CPU) and extract into the project as `onnxruntime-windows-x64-1.17.0`.

## 1) Get the code and model
1. Place this project folder somewhere like `C:\Projects\Part2`.
2. Ensure `yolov8n.onnx` is present in the project root (already included). If missing, see step 6 (optional).

## 2) Install Visual Studio C++ tools
1. Install "Visual Studio 2019/2022" (or "Build Tools for Visual Studio").
2. Make sure the "Desktop development with C++" workload is installed.
3. Open the "x64 Native Tools Command Prompt for VS".

## 3) Download ONNX Runtime (Windows)
1. Download ONNX Runtime (CPU, Windows x64) from the official releases.
2. Extract it into the project and rename the folder to:
   - `onnxruntime-windows-x64-1.17.0`

Your tree should contain:
```
Part2\onnxruntime-windows-x64-1.17.0\include
Part2\onnxruntime-windows-x64-1.17.0\lib
Part2\onnxruntime-windows-x64-1.17.0\bin
```

## 4) Easiest: One-click quick start (no OpenCV required)
Double‑click `quick_start_windows.bat` inside the project folder. It will:
- Detect MSVC tools
- Ensure ONNX Runtime DLLs are on PATH
- Build `test_basic.exe` and `simple_inference.exe`
- Run both tests automatically

If you prefer manual steps, see the batch file for the exact commands.

## 6) (Optional) Create a Python virtual environment (for model conversion)
Only needed if you want to (re)convert/download models with the Python tools under `models\`.
```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install ultralytics onnx opencv-python numpy

rem Convert/download a YOLOv8 model if needed
python models\convert_model.py
```

## 6) (Optional) Build the full application with OpenCV
To enable webcam/video processing and on-screen display, install OpenCV for Windows and link it.
1. Download OpenCV for Windows from the official site and extract to `C:\opencv`.
2. In the same VS command prompt:
```cmd
set OPENCV_DIR=C:\opencv\build
set PATH=%OPENCV_DIR%\x64\vc16\bin;%PATH%

rem Example compile (adjust the OpenCV library name to your version)
cl /std:c++17 /EHsc ^
  /I headers ^
  /I onnxruntime-windows-x64-1.17.0\include ^
  /I %OPENCV_DIR%\include ^
  src\main.cpp src\infer_engine.cpp src\preprocess.cpp src\nms.cpp src\frame_queue.cpp src\frame.cpp ^
  onnxruntime-windows-x64-1.17.0\lib\onnxruntime.lib ^
  %OPENCV_DIR%\x64\vc16\lib\opencv_world4xx.lib ^
  /Fe:inference_engine.exe
```

Run examples:
```cmd
inference_engine.exe --model yolov8n.onnx --video 0
inference_engine.exe --model yolov8n.onnx --video data\sample_video.mp4 --conf 0.3
```

## Notes
- Always start from the "x64 Native Tools Command Prompt for VS" so MSVC is available.
- Ensure ONNX Runtime DLL path is on `PATH` before running executables:
  - `set PATH=%CD%\onnxruntime-windows-x64-1.17.0\bin;%PATH%`
- You can delete the `venv\` folder before sending; steps above recreate it quickly when needed.

