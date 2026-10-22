
import json
import math
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ADAS Safety Copilot API")

PROJECT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_DIR / "environment_state.json"
DRIVER_FILE = PROJECT_DIR / "driver_state.json"

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


def read_json(path, defaults):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else defaults.copy()
    except (OSError, json.JSONDecodeError):
        return defaults.copy()


def clean_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def get_environment():
    raw = read_json(ENV_FILE, {
        "environment_risk": "UNKNOWN",
        "nearest_distance": None,
        "nearest_ttc": None,
        "nearest_class": "NONE",
        "nearest_id": -1,
    })

    return {
        **raw,
        "risk_level": raw.get(
            "environment_risk",
            raw.get("risk_level", "UNKNOWN"),
        ),
        "nearest_distance": clean_number(
            raw.get("nearest_distance")
        ),
        "ttc": clean_number(
            raw.get("nearest_ttc", raw.get("ttc"))
        ),
        "nearest_object": raw.get(
            "nearest_class",
            raw.get("nearest_object", "NONE"),
        ),
        "object_count": raw.get(
            "object_count",
            raw.get("detected_objects"),
        ),
    }


def get_driver():
    return read_json(DRIVER_FILE, {
        "driver_status": "UNKNOWN",
        "attention": None,
        "drowsiness": None,
        "head_status": "UNKNOWN",
        "eye_state": "UNKNOWN",
        "face_detected": False,
    })


def calculate_combined_risk(environment_risk, driver_status):
    # Mirrors the current combined_risk.py decision rules.
    environment_risk = str(environment_risk).upper()
    driver_status = str(driver_status).upper()

    if environment_risk == "CRITICAL":
        return "CRITICAL"

    if driver_status == "DROWSY":
        if environment_risk == "WARNING":
            return "CRITICAL"
        return "WARNING"

    if driver_status in ("INATTENTIVE", "DISTRACTED"):
        if environment_risk == "WARNING":
            return "CRITICAL"
        if environment_risk == "CAUTION":
            return "WARNING"
        return "CAUTION"

    return environment_risk


@app.get("/")
def root():
    return {
        "name": "ADAS Safety Copilot API",
        "status": "running",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "environment_file_found": ENV_FILE.exists(),
        "driver_file_found": DRIVER_FILE.exists(),
    }


@app.get("/api/environment")
def environment():
    return get_environment()


@app.get("/api/driver")
def driver():
    return get_driver()


@app.get("/api/combined")
def combined():
    env = get_environment()
    drv = get_driver()

    risk = calculate_combined_risk(
        env.get("risk_level", "UNKNOWN"),
        drv.get("driver_status", "UNKNOWN"),
    )

    return {
        "risk_level": risk,
        "environment_risk": env.get("risk_level"),
        "driver_status": drv.get("driver_status"),
        "message": f"Combined safety state: {risk}",
    }
