# ADAS Driver Safety Copilot

A CARLA-based Advanced Driver Assistance System (ADAS) prototype that combines environment perception with driver monitoring to estimate driving risk and display warnings.

> **Important status note:** The existing prototype has used RGB + LiDAR fusion. The intended low-cost version replaces LiDAR distance estimation with AI-based monocular depth estimation. Do not describe the system as fully LiDAR-free until that module has been integrated and validated.

## Features

- CARLA 0.9.16 simulation with a Tesla Model 3 ego vehicle.
- YOLO object detection from RGB camera frames.
- ByteTrack object tracking.
- Object distance/position and relative-motion estimation.
- Time-to-Collision (TTC) and collision-corridor risk assessment.
- Webcam driver monitoring using MediaPipe FaceMesh, head pose, and eye-state analysis.
- Fusion of driver state and environment risk into `SAFE`, `CAUTION`, `WARNING`, or `CRITICAL`.
- JSON state files used to communicate between processes.

## Architecture

```text
CARLA + Ego Tesla + NPC traffic
             |
             v
     RGB camera -> YOLO -> ByteTrack
             |                 |
             +---- Depth / distance estimation
                               |
                  Relative motion + TTC
                               |
                      Environment risk
                               |
Webcam -> FaceMesh -> Driver state
                               |
                               v
                         Risk fusion
                               |
                               v
                     Combined dashboard
```

## Requirements

- Windows 10/11
- CARLA 0.9.16
- Python 3.11 (previously tested with Python 3.11.9)
- Webcam for driver monitoring
- A compatible NVIDIA GPU can help, depending on the model and configuration

Python dependencies used by the scripts include `carla`, `ultralytics`, `opencv-python`, `numpy`, `pygame`, and `mediapipe==0.10.21`. Install any additional dependencies required by your current scripts. If the repository has a `requirements.txt`, use that file.

## Run the project

Open each step in a **separate terminal** and follow this order.

### 1. Start CARLA — Terminal 1

```powershell
cd C:\Users\avira\Downloads\CARLA_0.9.16
.\CarlaUE4.exe -quality-level=Low
```

Wait until the map has loaded.

### 2. Start the ego Tesla — Terminal 2

```powershell
cd C:\Users\avira\Downloads\CARLA_0.9.16\PythonAPI\examples
python manual_control.py --filter vehicle.tesla.model3 --res 640x360 --sync
```

Keep the ego vehicle at a controlled, low speed during testing.

### 3. Start NPC traffic — Terminal 3

```powershell
cd C:\Users\avira\Projects\carla_driver_project
python traffic.py
```

This runs the project's traffic-spawning script. Exact behavior depends on the current `traffic.py`.

### 4. Start the environment risk engine — Terminal 4

```powershell
cd C:\Users\avira\Projects\carla_driver_project
python risk_engine.py
```

The engine should connect to CARLA, find the ego vehicle, start its sensors, and process detections/risk. The exact console output depends on the current file version.

**Check the sensor message:** if the console says `LiDAR started`, that version still uses LiDAR. The camera-only depth replacement is a planned step unless you have already integrated it.

### 5. Start driver monitoring — Terminal 5

```powershell
cd C:\Users\avira\Projects\carla_driver_project
python driver_monitor.py
```

Allow the webcam to open and complete the neutral-face calibration. Keep your face visible and ensure the correct camera is selected.

### 6. Start combined risk dashboard — Terminal 6

```powershell
cd C:\Users\avira\Projects\carla_driver_project
python combined_risk.py
```

The dashboard reads the environment and driver state files and displays the combined safety state.

### Startup order summary

1. `CarlaUE4.exe`
2. `manual_control.py --sync`
3. `traffic.py`
4. `risk_engine.py`
5. `driver_monitor.py`
6. `combined_risk.py`

Do not run `sensors.py`, `live_perception.py`, or `fusion.py` alongside the final pipeline unless you are testing those components specifically. They are earlier/testing scripts and may consume extra resources or spawn additional sensors.

## Project files

| File | Purpose |
|---|---|
| `risk_engine.py` | Environment perception, tracking, distance/motion estimation, TTC and environment risk |
| `traffic.py` | Spawns surrounding NPC traffic |
| `driver_monitor.py` | Webcam-based face, head pose, eye state and drowsiness monitoring |
| `combined_risk.py` | Combines driver/environment states and displays final risk |
| `sensors.py` | Sensor capture/testing utility |
| `live_perception.py` | Earlier standalone YOLO test |
| `fusion.py` | Earlier RGB/depth fusion experiment |
| `yolo11n.pt` | YOLO model weights, if required by the script |

`manual_control.py` normally lives in CARLA's `PythonAPI\\examples` folder, not inside this project directory.

## Runtime state files

The scripts may generate:

- `environment_state.json` — environment risk and nearest-object information.
- `driver_state.json` — driver status, attention, drowsiness, and related fields.

These are generated runtime files and generally should not be committed to Git.

## Troubleshooting

### Ego vehicle not found

1. Start CARLA and wait for the map to load.
2. Start `manual_control.py` and confirm the Tesla is visible.
3. Test the CARLA connection:

```powershell
python -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(10); w=c.get_world(); print('Map:',w.get_map().name); vs=w.get_actors().filter('vehicle.*'); print('Vehicles:',len(vs)); [print(v.id, v.type_id, v.attributes.get('role_name')) for v in vs]"
```

### CARLA is not in synchronous mode

- Start `manual_control.py` with `--sync`.
- Make sure scripts connect to the same CARLA server at `127.0.0.1:2000`.
- Avoid running multiple scripts that independently change world settings or call `world.tick()`.

### MediaPipe `solutions` error

The previously used compatible version was:

```powershell
pip uninstall mediapipe -y
pip install mediapipe==0.10.21
```

Run the project using the same Python environment where dependencies were installed.

### Missing YOLO weights

If the script expects `yolo11n.pt`, provide the file or follow the model download instructions. Large model weights are often excluded from Git; document how to obtain them.

### Ego vehicle moves too fast

Use short throttle inputs and test at low speed. Do not assume an automatic speed limiter exists unless it has been implemented in the current code.

## Limitations and safety

- This is a simulation/research prototype, not a certified driver-assistance or autonomous-driving system.
- Monocular depth estimates require validation against ground truth before using them for TTC or safety decisions.
- Prototype TTC thresholds are not safety-certified.
- YOLO may miss or misclassify objects; LiDAR association in the earlier prototype may also fail with occlusion or sparse points.
- Webcam monitoring is affected by lighting, camera position, glasses, face angle, and occlusion.
- Physical steering/braking control and hardware-in-the-loop integration are future work.

## Progress and next steps

### Implemented or previously demonstrated

- CARLA connection and Tesla Model 3 ego vehicle.
- RGB camera and YOLO object detection.
- LiDAR-camera fusion and ByteTrack tracking in the existing prototype.
- Distance/position smoothing, relative-motion estimation, TTC, collision corridor and environment risk.
- Webcam driver monitoring using face landmarks, head pose and eye-state analysis.
- JSON communication between driver and environment processes.
- Combined risk dashboard.

### Planned next

1. Replace LiDAR distance estimation with a monocular depth model.
2. Validate estimated distance and TTC against CARLA ground truth across multiple scenarios.
3. Improve longitudinal TTC and warning stability.
4. Test cut-ins, sudden braking, lane departure, distraction and drowsiness.
5. Explore steering-wheel/pedal Hardware-in-the-Loop integration and edge deployment.

## Git notes

A `.gitignore` should exclude Python caches, virtual environments, secrets, generated sensor data, runtime JSON files, and large model weights. If model weights are needed to run the project, provide download instructions or use Git LFS/DVC.

## License and attribution

Add the license and required attribution for your own project and the libraries/models used before publishing the repository.
