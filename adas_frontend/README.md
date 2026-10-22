# ADAS Safety Copilot — React Dashboard

## Run
```powershell
npm install
npm run dev
```
Open the Vite URL, normally http://localhost:5173. Build with `npm run build`.

## Repository integration
The current Python project writes `environment_state.json` and `driver_state.json`. A browser cannot read arbitrary local files directly, so the dashboard starts in **Demo mode**. Turn Demo mode off in Settings to use a backend adapter.

Expected API endpoints:
- `GET /api/environment` — JSON from `environment_state.json`
- `GET /api/driver` — JSON from `driver_state.json`
- `GET /api/combined` — optional actual combined risk, e.g. `{"risk_level":"CAUTION","message":"Monitor the vehicle ahead"}`

Default API base: `http://127.0.0.1:8000`. Set `VITE_API_BASE` in a `.env` file to change the default. Enable CORS for `http://localhost:5173` in your backend.

The UI supports common field aliases, but verify actual keys emitted by your current scripts before relying on live values. For faithful final risk, expose the state produced by `combined_risk.py` rather than relying on frontend inference.

## Included
- Combined safety state banner
- TTC, nearest-object distance, object count and driver state metrics
- Risk trend chart
- Driver monitoring state
- Environment scene placeholder
- Recent event list and demo event generator
- API settings and responsive layout

## Not yet connected
This frontend does not directly connect to CARLA, display live RGB/depth frames, access the webcam, or modify the Python pipeline. Add an MJPEG/WebSocket/WebRTC stream endpoint to display actual camera frames. It is a prototype dashboard, not a certified safety system.
