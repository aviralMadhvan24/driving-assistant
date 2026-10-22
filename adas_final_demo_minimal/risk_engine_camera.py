"""
Camera-only ADAS environment risk engine.
CARLA RGB -> YOLO11n -> Depth Anything V2 Metric -> environment_state.json

Run after CARLA and manual_control.py.
Requires an ego vehicle with role_name="hero".

Prototype only:
- Depth Anything distances are estimates, not ground-truth measurements.
- TTC and risk levels are experimental and not certified for real vehicles.
- This script does not control brakes or steering.
"""

import json
import os
import time
from pathlib import Path

import carla
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ---------------- CONFIG ----------------
WIDTH, HEIGHT, FOV = 640, 360, 90
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
YOLO_PATH = "yolo11n.pt"
MAX_DEPTH_M = 80.0
PROCESS_EVERY_N_FRAMES = 3
CONFIDENCE = 0.30
STATE_PATH = Path(__file__).resolve().parent / "environment_state.json"

CAMERA_OFFSET = carla.Transform(
    carla.Location(x=0.8, y=0.0, z=2.4),
    carla.Rotation(pitch=-8.0),
)

# Demo thresholds only. Validate with known distances in CARLA.
CRITICAL_DISTANCE_M = 5.0
WARNING_DISTANCE_M = 12.0
CAUTION_DISTANCE_M = 25.0
CRITICAL_TTC_S = 1.5
WARNING_TTC_S = 3.0

# Camera-image collision corridor. This mirrors the two-line corridor
# from the previous risk engine, but uses image coordinates because this
# version does not use LiDAR. It is an approximation, not lane detection.
EGO_HALF_WIDTH = 0.93
SAFETY_MARGIN = 0.35
CORRIDOR_HALF_WIDTH_BOTTOM_PX = 125
CORRIDOR_HALF_WIDTH_TOP_PX = 38
CORRIDOR_TOP_Y = 45

# Only these detected classes are treated as road hazards for the headline risk.
# YOLO COCO IDs: person=0, bicycle=1, car=2, motorcycle=3, bus=5,
# train=6, truck=7, traffic light=9, stop sign=11.
HAZARD_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck"
}

# Match detections between frames by class and bounding-box IoU.
MIN_TRACK_IOU = 0.15
MIN_CLOSING_SPEED_MPS = 0.5
MAX_TTC_S = 30.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ENV] Inference device: {device}")
print("[ENV] Loading Depth Anything V2...")
processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME).to(device)
depth_model.eval()

print("[ENV] Loading YOLO...")
yolo = YOLO(YOLO_PATH)
yolo_device = 0 if device.type == "cuda" else "cpu"
print("[ENV] Models loaded.")

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(15)
world = client.get_world()


def find_ego():
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == "hero":
            return actor
    return None


ego = None
for _ in range(30):
    ego = find_ego()
    if ego is not None:
        break
    print("[ENV] Waiting for ego vehicle role_name='hero'...")
    time.sleep(1)

if ego is None:
    raise RuntimeError(
        "Ego vehicle not found. Start CARLA and manual_control.py first; "
        "the vehicle must have role_name='hero'."
    )

print(f"[ENV] Ego vehicle: {ego.id} ({ego.type_id})")

camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", str(WIDTH))
camera_bp.set_attribute("image_size_y", str(HEIGHT))
camera_bp.set_attribute("fov", str(FOV))
camera = world.spawn_actor(camera_bp, CAMERA_OFFSET, attach_to=ego)

latest_frame = None
latest_frame_id = -1
import threading
frame_lock = threading.Lock()


def camera_callback(image):
    global latest_frame, latest_frame_id
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        image.height, image.width, 4
    )
    bgr = bgra[:, :, :3].copy()
    with frame_lock:
        latest_frame = bgr
        latest_frame_id = image.frame


def estimate_depth(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = depth_model(**inputs)
        depth = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(HEIGHT, WIDTH),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    return depth.float().cpu().numpy()


def box_depth(depth_map, box):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(WIDTH - 1, x1))
    x2 = max(0, min(WIDTH, x2))
    y1 = max(0, min(HEIGHT - 1, y1))
    y2 = max(0, min(HEIGHT, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    dx, dy = x2 - x1, y2 - y1
    # Ignore box edges to reduce background pixels affecting the estimate.
    xa = x1 + int(dx * 0.20)
    xb = x2 - int(dx * 0.20)
    ya = y1 + int(dy * 0.20)
    yb = y2 - int(dy * 0.20)
    roi = depth_map[ya:yb, xa:xb]

    valid = roi[
        np.isfinite(roi) & (roi > 0.1) & (roi <= MAX_DEPTH_M)
    ]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


# Track state: track_id -> class_name, bbox, distance, monotonic timestamp.
tracks = {}
next_track_id = 1


def associate_and_update(detections, now):
    """Greedily associate same-class boxes with the best available IoU."""
    global next_track_id
    candidates = []

    for det_index, det in enumerate(detections):
        for track_id, old in tracks.items():
            if old["class_name"] != det["class_name"]:
                continue
            iou = box_iou(det["bbox"], old["bbox"])
            if iou >= MIN_TRACK_IOU:
                candidates.append((iou, det_index, track_id))

    candidates.sort(reverse=True)
    matched_dets = set()
    matched_tracks = set()
    assignments = {}

    for iou, det_index, track_id in candidates:
        if det_index in matched_dets or track_id in matched_tracks:
            continue
        matched_dets.add(det_index)
        matched_tracks.add(track_id)
        assignments[det_index] = track_id

    for index, det in enumerate(detections):
        track_id = assignments.get(index)
        if track_id is None:
            track_id = next_track_id
            next_track_id += 1
            det["ttc_seconds"] = None
        else:
            old = tracks[track_id]
            old_distance = old["distance_m"]
            dt = now - old["time"]

            ttc = None
            if (
                det["distance_m"] is not None
                and old_distance is not None
                and dt > 0.05
            ):
                # Smoothly reject implausibly large single-frame changes.
                closing_speed = (old_distance - det["distance_m"]) / dt
                if MIN_CLOSING_SPEED_MPS < closing_speed < 100.0:
                    estimated_ttc = det["distance_m"] / closing_speed
                    if 0 <= estimated_ttc <= MAX_TTC_S:
                        ttc = estimated_ttc
            det["ttc_seconds"] = (
                round(float(ttc), 2) if ttc is not None else None
            )

        det["track_id"] = track_id
        tracks[track_id] = {
            "class_name": det["class_name"],
            "bbox": det["bbox"],
            "distance_m": det["distance_m"],
            "time": now,
        }

    # Prevent the track dictionary growing indefinitely.
    stale_ids = [
        track_id for track_id, item in tracks.items()
        if now - item["time"] > 2.0
    ]
    for track_id in stale_ids:
        tracks.pop(track_id, None)


def corridor_bounds_at_y(y):
    """Return left/right image x-coordinates of the two-line corridor at y."""
    y = int(np.clip(y, CORRIDOR_TOP_Y, HEIGHT - 1))
    ratio = (y - CORRIDOR_TOP_Y) / max(1, (HEIGHT - 1 - CORRIDOR_TOP_Y))
    half_width = (
        CORRIDOR_HALF_WIDTH_TOP_PX
        + ratio * (
            CORRIDOR_HALF_WIDTH_BOTTOM_PX
            - CORRIDOR_HALF_WIDTH_TOP_PX
        )
    )
    center = WIDTH // 2
    return int(center - half_width), int(center + half_width)


def is_in_collision_corridor(bbox):
    """Approximate whether a detected object overlaps the forward corridor."""
    x1, y1, x2, y2 = bbox
    # Test the object's lower-middle point, closer to its road-contact region.
    sample_y = int(y1 + 0.80 * max(0, y2 - y1))
    left, right = corridor_bounds_at_y(sample_y)
    # Any horizontal overlap between the box and corridor counts as in-path.
    return x2 >= left and x1 <= right


def risk_for(distance, ttc):
    """Simple demo thresholds, not a certified collision prediction."""
    if ttc is not None and ttc <= CRITICAL_TTC_S:
        return "CRITICAL"
    if distance is not None and distance <= CRITICAL_DISTANCE_M:
        return "CRITICAL"
    if ttc is not None and ttc <= WARNING_TTC_S:
        return "WARNING"
    if distance is not None and distance <= WARNING_DISTANCE_M:
        return "WARNING"
    if distance is not None and distance <= CAUTION_DISTANCE_M:
        return "CAUTION"
    if distance is None:
        return "UNKNOWN"
    return "SAFE"


def risk_color(risk):
    return {
        "SAFE": (50, 190, 50),
        "CAUTION": (0, 200, 255),
        "WARNING": (0, 140, 255),
        "CRITICAL": (0, 0, 255),
        "UNKNOWN": (180, 180, 180),
    }.get(risk, (180, 180, 180))


def publish(state):
    temp = STATE_PATH.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(state, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp, STATE_PATH)


camera.listen(camera_callback)
print("[ENV] RGB camera attached to ego vehicle.")
print("[ENV] Press Q in the vision window to stop.")
last_frame_id = -1
processed = 0
last_log_time = 0.0

try:
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
            frame_id = latest_frame_id

        if frame is None or frame_id == last_frame_id:
            time.sleep(0.005)
            continue

        last_frame_id = frame_id
        if frame_id % PROCESS_EVERY_N_FRAMES != 0:
            continue

        depth_map = estimate_depth(frame)
        result = yolo.predict(
            frame,
            imgsz=640,
            conf=CONFIDENCE,
            device=yolo_device,
            verbose=False,
        )[0]

        annotated = frame.copy()
        detections = []
        now = time.monotonic()

        if result.boxes is not None:
            for box in result.boxes:
                coords = box.xyxy[0].detach().cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                class_id = int(box.cls[0].item())
                class_name = str(yolo.names[class_id])
                confidence = float(box.conf[0].item())
                distance = box_depth(depth_map, coords)

                detections.append({
                    "class_name": class_name,
                    "class_id": class_id,
                    "distance_m": round(distance, 2) if distance is not None else None,
                    "ttc_seconds": None,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(confidence, 3),
                })

        associate_and_update(detections, now)

        # Evaluate road hazards against the forward collision corridor.
        hazards = [
            det for det in detections
            if det["class_name"] in HAZARD_CLASSES
        ]
        for det in detections:
            det["in_corridor"] = (
                det["class_name"] in HAZARD_CLASSES
                and is_in_collision_corridor(det["bbox"])
            )
            det["risk_level"] = (
                risk_for(det["distance_m"], det["ttc_seconds"])
                if det["in_corridor"]
                else "SAFE"
            )

        in_path_hazards = [
            det for det in hazards if det["in_corridor"]
        ]
        measurable_in_path = [
            det for det in in_path_hazards
            if det["distance_m"] is not None
        ]
        nearest = (
            min(measurable_in_path, key=lambda det: det["distance_m"])
            if measurable_in_path else None
        )

        if nearest is None:
            # If a road hazard is in the corridor but depth failed, don't
            # claim SAFE: the risk cannot be assessed from this frame.
            overall_risk = "UNKNOWN" if in_path_hazards else "SAFE"
            nearest_distance = None
            nearest_ttc = None
            nearest_class = (
                in_path_hazards[0]["class_name"]
                if in_path_hazards else "NONE"
            )
            nearest_track_id = (
                in_path_hazards[0]["track_id"]
                if in_path_hazards else None
            )
        else:
            nearest_distance = nearest["distance_m"]
            nearest_ttc = nearest["ttc_seconds"]
            nearest_class = nearest["class_name"]
            nearest_track_id = nearest["track_id"]
            # Worst risk among objects in the corridor only.
            risk_rank = {
                "SAFE": 0, "CAUTION": 1, "WARNING": 2,
                "CRITICAL": 3, "UNKNOWN": -1
            }
            overall_risk = max(
                (det["risk_level"] for det in measurable_in_path),
                key=lambda value: risk_rank[value],
            )

        # Draw boxes. Risk coloring applies to road hazards.
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            if det["class_name"] in HAZARD_CLASSES:
                level = det["risk_level"]
            else:
                level = "SAFE"
            color = risk_color(level)
            distance_text = (
                f'{det["distance_m"]:.1f}m'
                if det["distance_m"] is not None else "depth N/A"
            )
            path_text = "PATH" if det["in_corridor"] else "SIDE"
            label = f'{det["class_name"]} {distance_text} {path_text} {det["confidence"]:.2f}'
            if det["ttc_seconds"] is not None:
                label += f' TTC {det["ttc_seconds"]:.1f}s'
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, label, (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

        # DRAW HIGH-VISIBILITY COLLISION CORRIDOR (two thick lines)
        # Draw after boxes so the guides cannot disappear behind detections.
        center_x = WIDTH // 2
        left_bottom = (center_x - CORRIDOR_HALF_WIDTH_BOTTOM_PX, HEIGHT - 2)
        left_top = (center_x - CORRIDOR_HALF_WIDTH_TOP_PX, CORRIDOR_TOP_Y)
        right_bottom = (center_x + CORRIDOR_HALF_WIDTH_BOTTOM_PX, HEIGHT - 2)
        right_top = (center_x + CORRIDOR_HALF_WIDTH_TOP_PX, CORRIDOR_TOP_Y)

        # Lightly tint the corridor, then draw a dark outline and bright yellow line.
        overlay = annotated.copy()
        corridor_poly = np.array([left_bottom, left_top, right_top, right_bottom], dtype=np.int32)
        cv2.fillPoly(overlay, [corridor_poly], (0, 220, 255))
        annotated = cv2.addWeighted(overlay, 0.10, annotated, 0.90, 0)
        cv2.line(annotated, left_bottom, left_top, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.line(annotated, right_bottom, right_top, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.line(annotated, left_bottom, left_top, (0, 255, 255), 4, cv2.LINE_AA)
        cv2.line(annotated, right_bottom, right_top, (0, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(annotated, "ESTIMATED COLLISION PATH", (center_x - 130, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2, cv2.LINE_AA)

        # High-visibility status HUD and collision-warning banners.
        banner_color = risk_color(overall_risk)
        cv2.rectangle(annotated, (0, 0), (WIDTH, 40), (15, 15, 15), -1)
        cv2.putText(annotated, f"ADAS RISK: {overall_risk}", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, banner_color, 2, cv2.LINE_AA)

        if overall_risk == "CRITICAL":
            cv2.rectangle(annotated, (35, 125), (WIDTH - 35, 205), (0, 0, 220), -1)
            cv2.rectangle(annotated, (35, 125), (WIDTH - 35, 205), (255, 255, 255), 2)
            cv2.putText(annotated, "!! COLLISION WARNING !!", (65, 158),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, "IMMEDIATE HAZARD IN ESTIMATED PATH", (70, 188),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        elif overall_risk == "WARNING":
            cv2.rectangle(annotated, (105, 135), (WIDTH - 105, 195), (0, 110, 255), -1)
            cv2.putText(annotated, "WARNING - HAZARD AHEAD", (125, 173),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        elif overall_risk == "CAUTION":
            cv2.rectangle(annotated, (190, 140), (450, 190), (0, 160, 220), -1)
            cv2.putText(annotated, "CAUTION", (250, 174),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        if nearest_distance is not None:
            detail = f"Nearest in-path: {nearest_class} | {nearest_distance:.1f} m"
            if nearest_ttc is not None:
                detail += f" | TTC {nearest_ttc:.1f} s"
        elif in_path_hazards:
            detail = "HAZARD IN PATH - DISTANCE UNAVAILABLE"
        else:
            detail = "No detected hazard in estimated path"
        cv2.rectangle(annotated, (0, HEIGHT - 30), (WIDTH, HEIGHT), (15, 15, 15), -1)
        cv2.putText(annotated, detail, (8, HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)

        state = {
            "source": "rgb_depth_anything_v2",
            "timestamp": time.time(),
            "frame_id": int(frame_id),
            "camera_fps": None,
            "environment_risk": overall_risk,
            "risk_level": overall_risk,
            "nearest_distance": nearest_distance,
            "nearest_ttc": nearest_ttc,
            "ttc": nearest_ttc,
            "nearest_class": nearest_class,
            "nearest_object": nearest_class,
            "nearest_id": nearest_track_id,
            "closing_speed": None,
            "object_count": len(detections),
            "hazard_count": len(hazards),
            "in_corridor_count": len(in_path_hazards),
            "collision_corridor": {
                "method": "approximate image-space trapezoid",
                "left_bottom_x": WIDTH // 2 - CORRIDOR_HALF_WIDTH_BOTTOM_PX,
                "right_bottom_x": WIDTH // 2 + CORRIDOR_HALF_WIDTH_BOTTOM_PX,
                "left_top_x": WIDTH // 2 - CORRIDOR_HALF_WIDTH_TOP_PX,
                "right_top_x": WIDTH // 2 + CORRIDOR_HALF_WIDTH_TOP_PX,
                "top_y": CORRIDOR_TOP_Y,
            },
            "detected_objects": detections,
            "depth_method": "Depth Anything V2 Metric Outdoor Small",
            "distance_unit": "meters",
            "risk_thresholds_demo": {
                "critical_distance_m": CRITICAL_DISTANCE_M,
                "warning_distance_m": WARNING_DISTANCE_M,
                "caution_distance_m": CAUTION_DISTANCE_M,
                "critical_ttc_seconds": CRITICAL_TTC_S,
                "warning_ttc_seconds": WARNING_TTC_S,
            },
        }

        publish(state)

        depth_visual = (
            np.clip(depth_map / MAX_DEPTH_M, 0, 1) * 255
        ).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_TURBO)
        cv2.putText(
            depth_colored, "AI depth visualization (0-80m scale)",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        # Separate full-size windows keep the corridor lines clearly visible.
        cv2.imshow("ADAS Risk Engine - COLLISION PATH", annotated)
        cv2.imshow("ADAS AI Depth Map", depth_colored)

        processed += 1
        if time.time() - last_log_time >= 1.0:
            print(
                f"[ENV] frame={frame_id} objects={len(detections)} "
                f"hazards={len(hazards)} in_path={len(in_path_hazards)} "
                f"nearest={nearest_distance}m "
                f"TTC={nearest_ttc} risk={overall_risk}"
            )
            last_log_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    try:
        camera.stop()
        camera.destroy()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print("[ENV] Camera risk engine stopped.")
