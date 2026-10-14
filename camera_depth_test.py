
import time
import threading

import carla
import cv2
import numpy as np
import torch

from PIL import Image
from ultralytics import YOLO
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)

# ---------------- CONFIG ----------------

WIDTH = 640
HEIGHT = 360
FOV = 90

DEPTH_MODEL_NAME = (
    "depth-anything/"
    "Depth-Anything-V2-Metric-Outdoor-Small-hf"
)

YOLO_PATH = "yolo11n.pt"
MAX_DISPLAY_DEPTH_M = 80.0
PROCESS_EVERY_N_FRAMES = 3

# ---------------- DEVICE ----------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Inference device:", device)

# ---------------- LOAD MODELS ----------------

print("Loading metric depth model...")
processor = AutoImageProcessor.from_pretrained(
    DEPTH_MODEL_NAME
)

depth_model = AutoModelForDepthEstimation.from_pretrained(
    DEPTH_MODEL_NAME
).to(device)

depth_model.eval()

print("Loading YOLO...")
yolo = YOLO(YOLO_PATH)

print("Models loaded.")

# ---------------- CARLA ----------------

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(15)

world = client.get_world()
blueprints = world.get_blueprint_library()

ego = next(
    (
        vehicle
        for vehicle in world.get_actors().filter("vehicle.*")
        if vehicle.attributes.get("role_name") == "hero"
    ),
    None,
)

if ego is None:
    raise RuntimeError(
        "Tesla with role_name='hero' not found. "
        "Start manual_control.py first."
    )

print("Ego vehicle:", ego.id, ego.type_id)

camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", str(WIDTH))
camera_bp.set_attribute("image_size_y", str(HEIGHT))
camera_bp.set_attribute("fov", str(FOV))

camera_transform = carla.Transform(
    carla.Location(x=0.8, y=0.0, z=2.4),
    carla.Rotation(pitch=-8.0),
)

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=ego,
)

# ---------------- FRAME CALLBACK ----------------

frame_lock = threading.Lock()
latest_frame = None
latest_frame_id = -1


def camera_callback(image):
    global latest_frame, latest_frame_id

    bgra = np.frombuffer(
        image.raw_data,
        dtype=np.uint8,
    ).reshape(image.height, image.width, 4)

    # CARLA supplies BGRA; OpenCV uses BGR.
    bgr = bgra[:, :, :3].copy()

    with frame_lock:
        latest_frame = bgr
        latest_frame_id = image.frame


# ---------------- DEPTH INFERENCE ----------------

def estimate_depth(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    inputs = processor(
        images=pil_image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = depth_model(**inputs)

        # Resize predicted depth to the original camera resolution.
        predicted = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(HEIGHT, WIDTH),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    return predicted.float().cpu().numpy()


# ---------------- OBJECT DEPTH ----------------

def box_depth(depth_map, box):
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(WIDTH - 1, x1))
    x2 = max(0, min(WIDTH, x2))
    y1 = max(0, min(HEIGHT - 1, y1))
    y2 = max(0, min(HEIGHT, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    # Use the inner part of the box to reduce edge/background effects.
    width = x2 - x1
    height = y2 - y1

    xa = x1 + int(width * 0.20)
    xb = x2 - int(width * 0.20)
    ya = y1 + int(height * 0.20)
    yb = y2 - int(height * 0.20)

    roi = depth_map[ya:yb, xa:xb]
    valid = roi[
        np.isfinite(roi)
        & (roi > 0.1)
        & (roi <= MAX_DISPLAY_DEPTH_M)
    ]

    if valid.size == 0:
        return None

    return float(np.median(valid))


# ---------------- MAIN LOOP ----------------

camera.listen(camera_callback)

print("RGB camera started. Press Q to exit.")

last_frame_id = -1
processed_count = 0

try:
    while True:
        with frame_lock:
            if latest_frame is None:
                frame = None
                frame_id = -1
            else:
                frame = latest_frame.copy()
                frame_id = latest_frame_id

        if frame is None or frame_id == last_frame_id:
            time.sleep(0.005)
            continue

        last_frame_id = frame_id

        if frame_id % PROCESS_EVERY_N_FRAMES != 0:
            continue

        depth_map = estimate_depth(frame)

        # Fixed display scale: 0 to 80 metres.
        depth_visual = (
            np.clip(
                depth_map / MAX_DISPLAY_DEPTH_M,
                0.0,
                1.0,
            ) * 255
        ).astype(np.uint8)

        depth_colored = cv2.applyColorMap(
            depth_visual,
            cv2.COLORMAP_TURBO,
        )

        # YOLO runs on RGB camera frames, not on LiDAR.
        yolo_device = 0 if torch.cuda.is_available() else "cpu"

        results = yolo.predict(
            frame,
            imgsz=640,
            conf=0.35,
            device=yolo_device,
            verbose=False,
        )

        annotated = frame.copy()

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].item())
                class_name = yolo.names[class_id]

                distance = box_depth(depth_map, coords)

                x1, y1, x2, y2 = map(int, coords)

                if distance is None:
                    label = f"{class_name}: depth unavailable"
                else:
                    label = f"{class_name}: ~{distance:.1f} m"

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        cv2.putText(
            depth_colored,
            "Estimated depth (display: 0-80 m)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        combined = np.hstack((annotated, depth_colored))
        cv2.imshow("RGB detections | AI depth", combined)

        processed_count += 1

        if processed_count % 10 == 0:
            print("Processed frames:", processed_count)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.stop()
    camera.destroy()
    cv2.destroyAllWindows()
    print("Depth test stopped.")
