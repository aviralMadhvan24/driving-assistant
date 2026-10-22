"""FastAPI bridge for the camera-only CARLA ADAS demo.

Reads environment_state.json from risk_engine_camera.py, exposes CARLA ego
vehicle telemetry, and serves the latest annotated perception frame if the
risk engine is configured to write annotated_frame.jpg.

Driver monitoring is intentionally marked unavailable while the physical
webcam is not working.
"""
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / "environment_state.json"
DRIVER_FILE = ROOT / "driver_state.json"  # retained for later; not trusted while camera is off
ANNOTATED_IMAGE = ROOT / "annotated_frame.jpg"

app = FastAPI(title="ADAS Safety Copilot API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

def read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else fallback.copy()
    except (OSError, json.JSONDecodeError):
        return fallback.copy()

def state_age(state: dict[str, Any]) -> float | None:
    try:
        timestamp = float(state["timestamp"])
        age = time.time() - timestamp
        return max(0.0, age)
    except (KeyError, TypeError, ValueError):
        return None

def with_freshness(state: dict[str, Any]) -> dict[str, Any]:
    result = state.copy()
    age = state_age(state)
    result["data_age_seconds"] = round(age, 2) if age is not None else None
    result["source_status"] = (
        "live" if age is not None and age <= 5 else
        "stale" if age is not None else
        "unavailable"
    )
    # A missing/stale perception file must never be presented as a current SAFE.
    if result["source_status"] != "live":
        result["risk_level"] = "UNKNOWN"
        result["environment_risk"] = "UNKNOWN"
    return result

def get_carla_telemetry() -> dict[str, Any]:
    """Read-only connection to CARLA. API still runs if CARLA is unavailable."""
    try:
        import carla  # Imported here so API can start without the CARLA package.
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(1.5)
        world = client.get_world()
        ego = None
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name") == "hero":
                ego = actor
                break
        if ego is None:
            return {
                "source_status": "unavailable",
                "message": "No vehicle with role_name='hero' found",
                "map": world.get_map().name,
                "ego_vehicle": None,
            }

        velocity = ego.get_velocity()
        speed_mps = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
        transform = ego.get_transform()
        location = transform.location
        rotation = transform.rotation
        return {
            "source_status": "live",
            "timestamp": time.time(),
            "map": world.get_map().name,
            "ego_vehicle": {
                "id": ego.id,
                "type": ego.type_id,
                "role_name": ego.attributes.get("role_name", ""),
                "speed_mps": round(speed_mps, 2),
                "speed_kmh": round(speed_mps * 3.6, 1),
                "position": {
                    "x": round(location.x, 2),
                    "y": round(location.y, 2),
                    "z": round(location.z, 2),
                },
                "heading_degrees": round(rotation.yaw, 1),
            },
            "vehicle_count": len(world.get_actors().filter("vehicle.*")),
            "pedestrian_count": len(world.get_actors().filter("walker.pedestrian.*")),
        }
    except Exception as exc:
        return {
            "source_status": "unavailable",
            "message": f"CARLA telemetry unavailable: {type(exc).__name__}: {exc}",
            "ego_vehicle": None,
        }

@app.get("/")
def root():
    return {"service": "ADAS Safety Copilot API", "docs": "/docs"}

@app.get("/api/health")
def health():
    env = read_json(ENV_FILE, {})
    age = state_age(env)
    return {
        "status": "ok",
        "environment_file_exists": ENV_FILE.exists(),
        "driver_file_exists": DRIVER_FILE.exists(),
        "annotated_image_exists": ANNOTATED_IMAGE.exists(),
        "environment_source_status": (
            "live" if age is not None and age <= 5 else
            "stale" if age is not None else "unavailable"
        ),
        "timestamp": time.time(),
    }

@app.get("/api/environment")
def environment():
    state = read_json(ENV_FILE, {})
    if not state:
        return {
            "source_status": "unavailable",
            "risk_level": "UNKNOWN",
            "environment_risk": "UNKNOWN",
            "nearest_distance": None,
            "nearest_ttc": None,
            "nearest_class": "NONE",
            "object_count": 0,
            "detected_objects": [],
        }
    return with_freshness(state)

@app.get("/api/driver")
def driver():
    # Physical driver camera is currently unavailable. Do not expose old JSON
    # as if it were a live reading.
    return {
        "source_status": "unavailable",
        "driver_status": "UNAVAILABLE",
        "attention": None,
        "drowsiness": None,
        "face_detected": False,
        "message": "Driver camera disabled until the webcam is working.",
    }

@app.get("/api/combined")
def combined():
    env = read_json(ENV_FILE, {})
    age = state_age(env)
    env_live = age is not None and age <= 5
    env_risk = env.get("environment_risk", env.get("risk_level", "UNKNOWN")) if env_live else "UNKNOWN"
    # No driver input is used while the driver camera is unavailable.
    return {
        "combined_risk": env_risk,
        "environment_risk": env_risk,
        "driver_status": "UNAVAILABLE",
        "environment_data_age_seconds": round(age, 2) if age is not None else None,
        "driver_data_age_seconds": None,
        "source_status": "live" if env_live else "stale_or_missing",
        "note": "Combined risk currently reflects environment perception only; driver monitoring is offline.",
    }

@app.get("/api/telemetry")
def telemetry():
    return get_carla_telemetry()

@app.get("/api/perception/image")
def perception_image():
    if not ANNOTATED_IMAGE.exists():
        raise HTTPException(
            status_code=404,
            detail="Annotated frame not found. Add the image-write patch to risk_engine_camera.py and restart it.",
        )
    age = time.time() - ANNOTATED_IMAGE.stat().st_mtime
    if age > 5:
        raise HTTPException(status_code=503, detail="Annotated perception frame is stale.")
    return FileResponse(
        ANNOTATED_IMAGE,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )
