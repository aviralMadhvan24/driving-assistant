# ADAS Safety Copilot — minimal demo files

This folder intentionally keeps the demo to three Python files:

1. `risk_engine_camera.py` — CARLA RGB + YOLO11n + Depth Anything V2; writes `environment_state.json`.
2. `driver_monitor.py` — webcam face/head/eye monitoring; writes `driver_state.json`.
3. `main.py` — FastAPI API bridge for the React dashboard.

Keep your existing React `frontend/` folder separately. This bundle does not include the frontend.

## Install dependencies

Use the same Python environment in which CARLA, PyTorch/CUDA, YOLO and Depth Anything V2 already work. Install any missing packages:

```powershell
python -m pip install -r requirements.txt
```

CARLA's Python API (`carla`) may need to come from the PythonAPI/egg included with your CARLA installation if pip cannot install it.

## Start in separate PowerShell terminals

### 1. Start CARLA

```powershell
cd C:\Users\avira\Downloads\CARLA_0.9.16
.\CarlaUE4.exe -quality-level=Low
```

### 2. Start the ego car

```powershell
cd C:\Users\avira\Downloads\CARLA_0.9.16\PythonAPI\examples
python manual_control.py --filter vehicle.tesla.model3 --res 640x360 --sync
```

### 3. Start camera-only environment risk engine

Run from this folder so the JSON file is created beside the scripts:

```powershell
cd C:\Users\avira\Projects\carla_driver_project
python risk_engine_camera.py
```

This uses the CARLA RGB camera, YOLO11n, and Depth Anything V2. Press Q in the vision window to stop it.

### 4. Start driver monitor

```powershell
cd C:\Users\avira\Projects\carla_driver_project
python driver_monitor.py
```

Allow camera permission and look straight at the webcam during calibration. Press Q to stop.

### 5. Start API

```powershell
cd C:\Users\avira\Projects\carla_driver_project
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Check:
- http://127.0.0.1:8000/api/health
- http://127.0.0.1:8000/api/environment
- http://127.0.0.1:8000/api/driver
- http://127.0.0.1:8000/api/combined

### 6. Start existing React frontend

In a separate terminal, from your existing frontend directory:

```powershell
npm install
npm run dev
```

Set `VITE_API_BASE=http://127.0.0.1:8000` in `frontend/.env` if your frontend uses that variable, then restart Vite. In its settings, turn Demo Mode off.

## Important demo limitations

- Depth estimates and risk thresholds are experimental; validate against CARLA ground truth before presenting them as accurate.
- TTC is a rough estimate from detections across frames and is not reliable object tracking. Treat it as a demo metric, not a safety-certified measurement.
- YOLO detects only the classes in its pretrained model; it will not reliably detect potholes or animals without suitable model support.
- The API reports stale/missing data as UNKNOWN instead of falsely declaring SAFE.
- Do not run the older `risk_engine.py`, `risk_engine_sync.py`, `risk_engine_final.py`, or `combined_risk.py` in parallel with this pipeline.
