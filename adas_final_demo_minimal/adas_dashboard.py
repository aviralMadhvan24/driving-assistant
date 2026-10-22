import json
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh

try:
    import carla
except Exception:
    carla = None

BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / "environment_state.json"
FRAME_PATH = BASE_DIR / "annotated_frame.jpg"

st.set_page_config(
    page_title="ADAS Safety Copilot",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1.5rem;}
[data-testid="stMetric"] {background: #111827; padding: 14px 16px; border: 1px solid #263246; border-radius: 12px;}
.risk {padding: 18px 22px; border-radius: 12px; margin-bottom: 12px; border: 1px solid #334155;}
.risk-safe {background:#10251d; border-color:#24734b;}
.risk-caution {background:#2a2415; border-color:#9a7625;}
.risk-warning {background:#2b1d12; border-color:#b45b20;}
.risk-critical {background:#301719; border-color:#b8323c;}
.risk-unknown {background:#171d29; border-color:#475569;}
.small-muted {color:#94a3b8; font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)


def read_state():
    """Read JSON with retries to tolerate a brief Windows write window."""
    if not STATE_PATH.exists():
        return None, "environment_state.json not found"
    last_error = None
    for _ in range(4):
        try:
            with STATE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None, "Environment state is not a JSON object"
            return data, None
        except (OSError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(0.05)
    return None, f"Could not read environment state: {last_error}"


def number(value, digits=1, suffix=""):
    try:
        if value is None:
            return "—"
        value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            return "—"
        return f"{value:.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def age_seconds(state):
    try:
        return max(0.0, time.time() - float(state.get("timestamp")))
    except (TypeError, ValueError):
        return None


def get_telemetry():
    """Read-only CARLA telemetry. No vehicle controls are sent."""
    if carla is None:
        return None, "CARLA Python API import failed in this Python environment"
    client = None
    try:
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(1.5)
        world = client.get_world()
        vehicles = world.get_actors().filter("vehicle.*")
        ego = None
        for vehicle in vehicles:
            if vehicle.attributes.get("role_name") == "hero":
                ego = vehicle
                break
        if ego is None:
            return {
                "map": world.get_map().name,
                "vehicle_count": len(vehicles),
                "pedestrian_count": len(world.get_actors().filter("walker.pedestrian.*")),
            }, "Connected to CARLA, but no vehicle with role_name='hero' was found"

        velocity = ego.get_velocity()
        speed_ms = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
        transform = ego.get_transform()
        return {
            "map": world.get_map().name,
            "speed_kmh": speed_ms * 3.6,
            "speed_ms": speed_ms,
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z,
            "yaw": transform.rotation.yaw,
            "vehicle_count": len(vehicles),
            "pedestrian_count": len(world.get_actors().filter("walker.pedestrian.*")),
            "vehicle_id": ego.id,
            "vehicle_type": ego.type_id,
        }, None
    except Exception as exc:
        return None, f"CARLA telemetry unavailable: {exc}"


st_autorefresh(interval=1000, key="adas_live_refresh")

st.title("🚘 ADAS Safety Copilot")
st.caption("Python-only live dashboard · CARLA simulation prototype · no vehicle control is issued")

with st.sidebar:
    st.header("System status")
    refresh_seconds = st.slider("Refresh interval (seconds)", 1, 5, 1)
    st.markdown("---")
    st.markdown("**Data sources**")
    st.write("🟡 Environment perception: `environment_state.json`")
    st.write("🟡 Annotated frame: `annotated_frame.jpg`")
    st.write("⚪ Driver camera: unavailable")
    st.markdown("---")
    st.caption("Prototype only. Not certified for real-world driving or safety decisions.")

# Streamlit reruns the app at this interval through its native fragment API where available.
# For compatibility, the page can also be refreshed using the button below.
if st.button("↻ Refresh now", use_container_width=True):
    st.rerun()

state, state_error = read_state()
telemetry, telemetry_error = get_telemetry()

if state is None:
    risk = "UNKNOWN"
    age = None
    st.warning(state_error or "Waiting for the perception engine to publish state.")
else:
    age = age_seconds(state)
    risk = str(state.get("risk_level") or state.get("environment_risk") or "UNKNOWN").upper()
    if age is not None and age > 5:
        risk = "UNKNOWN"
        st.warning(f"Environment state is stale ({age:.1f} seconds old). Risk is shown as UNKNOWN.")

risk_class = risk.lower() if risk in {"SAFE", "CAUTION", "WARNING", "CRITICAL"} else "unknown"
risk_message = {
    "SAFE": "No in-corridor hazard reported by the current perception pipeline.",
    "CAUTION": "Caution: monitor the scene and detected objects.",
    "WARNING": "Warning: the perception pipeline reports elevated risk.",
    "CRITICAL": "Critical: the perception pipeline reports its highest risk level.",
}.get(risk, "Waiting for fresh, valid environment data.")

st.markdown(
    f'<div class="risk risk-{risk_class}"><div class="small-muted">ENVIRONMENT RISK</div>'
    f'<h2 style="margin:3px 0 5px 0">{risk}</h2><div>{risk_message}</div></div>',
    unsafe_allow_html=True,
)

if state:
    object_count = state.get("object_count")
    detected = state.get("detected_objects")
    if object_count is None and isinstance(detected, list):
        object_count = len(detected)

    nearest_class = state.get("nearest_object") or state.get("nearest_class")
    nearest_distance = state.get("nearest_distance")
    ttc = state.get("ttc", state.get("nearest_ttc"))
    frame_id = state.get("frame_id")
    state_source = state.get("source", "unknown")
else:
    object_count = nearest_class = nearest_distance = ttc = frame_id = None
    state_source = "unknown"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Nearest object", str(nearest_class or "—"))
m2.metric("Estimated distance", number(nearest_distance, 1, " m"))
m3.metric("Time to collision", number(ttc, 1, " s"))
m4.metric("Detected objects", str(object_count if object_count is not None else "—"))

left, right = st.columns([1.35, 1])
with left:
    st.subheader("Live perception camera")
    if FRAME_PATH.exists():
        frame_age = time.time() - FRAME_PATH.stat().st_mtime
        if frame_age <= 5:
            st.image(str(FRAME_PATH), use_container_width=True)
            st.caption(f"Annotated frame · {frame_age:.1f}s old")
        else:
            st.warning(f"Annotated frame is stale ({frame_age:.1f}s old). Restart/check the camera risk engine.")
            st.image(str(FRAME_PATH), use_container_width=True)
    else:
        st.info("Waiting for annotated_frame.jpg. Make sure the updated camera engine saves this file.")
    st.caption(f"Perception source: {state_source} · Frame: {frame_id if frame_id is not None else '—'}")

with right:
    st.subheader("CARLA vehicle telemetry")
    if telemetry:
        a, b = st.columns(2)
        a.metric("Ego speed", number(telemetry.get("speed_kmh"), 1, " km/h"))
        b.metric("Map", str(telemetry.get("map", "—")))
        c, d = st.columns(2)
        c.metric("Vehicles", str(telemetry.get("vehicle_count", "—")))
        d.metric("Pedestrians", str(telemetry.get("pedestrian_count", "—")))
        if telemetry.get("vehicle_id") is not None:
            st.caption(f"Ego: {telemetry.get('vehicle_type')} · ID {telemetry.get('vehicle_id')}")
            st.write({
                "Position X": number(telemetry.get("x"), 2, " m"),
                "Position Y": number(telemetry.get("y"), 2, " m"),
                "Position Z": number(telemetry.get("z"), 2, " m"),
                "Heading": number(telemetry.get("yaw"), 1, "°"),
            })
    if telemetry_error:
        st.warning(telemetry_error)

st.subheader("Detected objects")
objects = state.get("detected_objects", []) if state else []
if isinstance(objects, list) and objects:
    rows = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        rows.append({
            "Class": obj.get("class_name", obj.get("class", obj.get("label", "Unknown"))),
            "Confidence": obj.get("confidence"),
            "Distance (m)": obj.get("distance_m", obj.get("distance")),
            "In corridor": obj.get("in_corridor", obj.get("in_path", "—")),
            "Track ID": obj.get("track_id", obj.get("id", "—")),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Object entries exist but do not match the expected dictionary format.")
else:
    st.info("No object list is available in the current state file. This can happen if objects are detected but none qualify as tracked hazards, or if the state file is from an older engine version.")

st.subheader("Driver monitoring")
st.info("UNAVAILABLE — the webcam/driver-monitoring pipeline is currently offline. No driver attention or drowsiness values are being fabricated.")

with st.expander("Raw environment JSON"):
    if state:
        st.json(state)
    else:
        st.code(state_error or "No state available")

if state:
    timestamp = state.get("timestamp")
    try:
        timestamp_text = datetime.fromtimestamp(float(timestamp)).strftime("%H:%M:%S")
    except (TypeError, ValueError, OSError):
        timestamp_text = "not provided"
    st.caption(
        f"Last environment update: {timestamp_text} · "
        f"State age: {number(age, 1, ' s')} · "
        "Dashboard reads state only; it does not control CARLA."
    )

st.caption("Dashboard auto-refreshes every second. Use Refresh now for an immediate rerun.")
