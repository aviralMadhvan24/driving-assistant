import cv2
import json
import os
import time
import numpy as np

DRIVER_STATE_FILE = "driver_state.json"
ENVIRONMENT_STATE_FILE = "environment_state.json"
UPDATE_INTERVAL = 0.05

COLORS = {
    "SAFE": (0, 220, 0),
    "CAUTION": (0, 220, 255),
    "WARNING": (0, 140, 255),
    "CRITICAL": (0, 0, 255)
}

DEFAULT_DRIVER_STATE = {
    "driver_status": "INATTENTIVE",
    "attention": 0.0,
    "drowsiness": 100.0,
    "head_status": "INATTENTIVE",
    "eye_state": "UNKNOWN",
    "face_detected": False
}

DEFAULT_ENVIRONMENT_STATE = {
    "environment_risk": "SAFE",
    "nearest_distance": float("inf"),
    "nearest_ttc": None,
    "nearest_class": "NONE",
    "nearest_id": -1
}


def read_state(filename, default_state):
    if not os.path.exists(filename):
        return default_state.copy()
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except Exception:
        return default_state.copy()


def calculate_combined_risk(environment_risk, driver_status):
    if environment_risk == "CRITICAL":
        return "CRITICAL"
    if driver_status == "DROWSY":
        if environment_risk == "WARNING":
            return "CRITICAL"
        if environment_risk == "CAUTION":
            return "WARNING"
        return "WARNING"
    if driver_status == "INATTENTIVE":
        if environment_risk == "WARNING":
            return "CRITICAL"
        if environment_risk == "CAUTION":
            return "WARNING"
        return "CAUTION"
    if driver_status == "DISTRACTED":
        if environment_risk == "WARNING":
            return "CRITICAL"
        if environment_risk == "CAUTION":
            return "WARNING"
        return "CAUTION"
    return environment_risk


def draw_text(frame, text, position, scale=0.62, thickness=2, color=(255, 255, 255)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


print("=" * 65)
print("COMBINED DRIVER SAFETY RISK ENGINE")
print("=" * 65)
print("Reading driver_state.json and environment_state.json")
print("Press Q to quit.")
print("=" * 65)

while True:
    driver = read_state(DRIVER_STATE_FILE, DEFAULT_DRIVER_STATE)
    environment = read_state(ENVIRONMENT_STATE_FILE, DEFAULT_ENVIRONMENT_STATE)

    driver_status = driver.get("driver_status", "INATTENTIVE")
    attention = float(driver.get("attention", 0.0))
    drowsiness = float(driver.get("drowsiness", 100.0))
    head_status = driver.get("head_status", "INATTENTIVE")
    eye_state = driver.get("eye_state", "UNKNOWN")
    face_detected = driver.get("face_detected", False)

    environment_risk = environment.get("environment_risk", "SAFE")
    distance = environment.get("nearest_distance", float("inf"))
    ttc = environment.get("nearest_ttc", None)
    nearest_class = environment.get("nearest_class", "NONE")
    nearest_id = environment.get("nearest_id", -1)

    try:
        distance = float(distance)
    except Exception:
        distance = float("inf")

    try:
        ttc = float(ttc) if ttc is not None else float("inf")
    except Exception:
        ttc = float("inf")

    final_risk = calculate_combined_risk(environment_risk, driver_status)
    color = COLORS.get(final_risk, (255, 255, 255))

    frame = np.zeros((500, 900, 3), dtype=np.uint8)
    frame[:] = (25, 25, 25)

    draw_text(frame, "AI DRIVER SAFETY SYSTEM", (30, 45), 0.95, 3)

    draw_text(frame, "ENVIRONMENT PERCEPTION", (30, 95), 0.72, 2)
    draw_text(frame, f"Risk: {environment_risk}", (30, 130), color=COLORS.get(environment_risk, (255, 255, 255)))
    draw_text(frame, f"Distance: {'---' if np.isinf(distance) else f'{distance:.1f} m'}", (30, 165))
    draw_text(frame, f"TTC: {'INF' if np.isinf(ttc) else f'{ttc:.1f} s'}", (30, 200))
    draw_text(frame, f"Object: {nearest_class}", (30, 235))
    draw_text(frame, f"Track ID: {nearest_id}", (30, 270))

    draw_text(frame, "DRIVER MONITORING", (470, 95), 0.72, 2)
    draw_text(frame, f"Status: {driver_status}", (470, 130))
    draw_text(frame, f"Attention: {attention:.0f}%", (470, 165))
    draw_text(frame, f"Drowsiness: {drowsiness:.0f}%", (470, 200))
    draw_text(frame, f"Head: {head_status}", (470, 235))
    draw_text(frame, f"Eyes: {eye_state}", (470, 270))
    draw_text(frame, f"Face: {'DETECTED' if face_detected else 'NOT DETECTED'}", (470, 305))

    cv2.rectangle(frame, (30, 350), (870, 455), color, 4)
    draw_text(frame, "FINAL DRIVER RISK", (55, 385), 0.72, 2)
    draw_text(frame, final_risk, (55, 435), 1.2, 4, color)

    if final_risk == "CRITICAL":
        draw_text(frame, "!!! IMMEDIATE ATTENTION REQUIRED !!!", (400, 435), 0.58, 3, color)
    elif final_risk == "WARNING":
        draw_text(frame, "WARNING - DRIVER ATTENTION REQUIRED", (400, 435), 0.48, 2, color)
    elif final_risk == "CAUTION":
        draw_text(frame, "CAUTION", (400, 435), 0.62, 2, color)
    else:
        draw_text(frame, "DRIVING CONDITIONS NORMAL", (400, 435), 0.48, 2, color)

    cv2.imshow("Combined Driver Safety System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    time.sleep(UPDATE_INTERVAL)

cv2.destroyAllWindows()
print("Combined risk engine stopped.")
