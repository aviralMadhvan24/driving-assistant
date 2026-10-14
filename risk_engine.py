import carla

import cv2

import numpy as np

import math
import json
import os
import time

from collections import deque

from ultralytics import YOLO

# ============================================================

# CONFIG

# ============================================================

WIDTH = 640

HEIGHT = 360

FOV = 90

LIDAR_RANGE = 50.0

DISTANCE_ALPHA = 0.15

POSITION_ALPHA = 0.30

VELOCITY_HISTORY_SIZE = 6

MIN_CLOSING_SPEED = 0.5

TTC_CRITICAL = 1.5

TTC_WARNING = 3.0

TTC_CAUTION = 6.0

# ------------------------------------------------------------

# EGO VEHICLE / COLLISION CORRIDOR

# ------------------------------------------------------------

# Tesla Model 3 is approximately 1.85m wide.

EGO_HALF_WIDTH = 0.93

# Extra safety margin on both sides.

SAFETY_MARGIN = 0.35

# Maximum corridor width.

MAX_CORRIDOR_HALF_WIDTH = 2.0

# ------------------------------------------------------------

# DISTANCE FALLBACK

# ------------------------------------------------------------

CRITICAL_DISTANCE = 4.0

WARNING_DISTANCE = 8.0

CAUTION_DISTANCE = 15.0

MODEL_PATH = "yolo11n.pt"

ENVIRONMENT_STATE_FILE = "environment_state.json"

# Synchronized sensor processing
SENSOR_BUFFER_SIZE = 30
SYNC_WAIT_SLEEP = 0.001
REQUIRE_SYNCHRONOUS_MODE = True


def publish_environment_state(
    overall_risk,
    nearest_distance,
    nearest_ttc,
    nearest_class,
    nearest_id
):
    state = {
        "environment_risk": overall_risk,
        "nearest_distance": (
            None
            if math.isinf(nearest_distance)
            else float(nearest_distance)
        ),
        "nearest_ttc": (
            None
            if math.isinf(nearest_ttc)
            else float(nearest_ttc)
        ),
        "nearest_class": nearest_class,
        "nearest_id": int(nearest_id)
    }

    temp_file = ENVIRONMENT_STATE_FILE + ".tmp"

    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(state, f)

        os.replace(
            temp_file,
            ENVIRONMENT_STATE_FILE
        )

    except Exception:
        pass


# ============================================================

# GLOBALS

# ============================================================

latest_rgb = None

latest_lidar = None

latest_lidar_transform = None

latest_lidar_timestamp = None

rgb_frames = {}
lidar_frames = {}
last_processed_frame = -1

filtered_distances = {}

filtered_positions = {}

track_history = {}

# ============================================================

# YOLO

# ============================================================

model = YOLO(MODEL_PATH)

# ============================================================

# CARLA CONNECTION

# ============================================================

client = carla.Client(

    "127.0.0.1",

    2000

)

client.set_timeout(10.0)

world = client.get_world()

blueprint_library = (

    world.get_blueprint_library()

)

# ============================================================

# FIND EGO VEHICLE

# ============================================================

# ============================================================
# FIND EGO VEHICLE
# ============================================================

# manual_control.py may need a short moment to spawn the vehicle.
# Wait instead of failing immediately.

EGO_WAIT_TIMEOUT = 15.0
EGO_POLL_INTERVAL = 0.25

ego_vehicle = None
ego_deadline = time.time() + EGO_WAIT_TIMEOUT

print("Waiting for ego vehicle...")

while time.time() < ego_deadline:

    vehicles = world.get_actors().filter(
        "vehicle.*"
    )

    # Prefer the vehicle marked as hero.
    for vehicle in vehicles:

        if vehicle.attributes.get(
            "role_name"
        ) == "hero":

            ego_vehicle = vehicle
            break

    # Fallback to Tesla Model 3.
    if ego_vehicle is None:

        for vehicle in vehicles:

            if vehicle.type_id == (
                "vehicle.tesla.model3"
            ):

                ego_vehicle = vehicle
                break

    if ego_vehicle is not None:
        break

    time.sleep(EGO_POLL_INTERVAL)

if ego_vehicle is None:

    raise RuntimeError(
        "Ego vehicle was not found after "
        f"{EGO_WAIT_TIMEOUT:.0f} seconds. "
        "Make sure manual_control.py is running "
        "and the vehicle is spawned."
    )

print(
    "Ego:",
    ego_vehicle.type_id,
    "ID:",
    ego_vehicle.id,
    "role:",
    ego_vehicle.attributes.get("role_name")
)

# ------------------------------------------------------------
# SYNCHRONOUS MODE CHECK
# ------------------------------------------------------------

world_settings = world.get_settings()

if REQUIRE_SYNCHRONOUS_MODE:

    if not world_settings.synchronous_mode:

        raise RuntimeError(
            "CARLA is not in synchronous mode. "
            "Start manual_control.py with --sync "
            "before starting risk_engine_sync.py."
        )

    print("Synchronous mode: ON")

else:

    print(
        "Synchronous mode:",
        "ON" if world_settings.synchronous_mode else "OFF"
    )

# ============================================================

# CAMERA

# ============================================================

camera_bp = blueprint_library.find(

    "sensor.camera.rgb"

)

camera_bp.set_attribute(

    "image_size_x",

    str(WIDTH)

)

camera_bp.set_attribute(

    "image_size_y",

    str(HEIGHT)

)

camera_bp.set_attribute(

    "fov",

    str(FOV)

)

camera_transform = carla.Transform(

    carla.Location(

        x=0.8,

        y=0.0,

        z=2.4

    ),

    carla.Rotation(

        pitch=-8.0,

        yaw=0.0,

        roll=0.0

    )

)

camera = world.spawn_actor(

    camera_bp,

    camera_transform,

    attach_to=ego_vehicle

)

# ============================================================

# LIDAR

# ============================================================

lidar_bp = blueprint_library.find(

    "sensor.lidar.ray_cast"

)

lidar_bp.set_attribute(

    "range",

    str(LIDAR_RANGE)

)

lidar_bp.set_attribute(

    "channels",

    "16"

)

lidar_bp.set_attribute(

    "points_per_second",

    "30000"

)

lidar_bp.set_attribute(

    "rotation_frequency",

    "20"

)

lidar_transform = carla.Transform(

    carla.Location(

        x=0.0,

        y=0.0,

        z=2.2

    )

)

lidar = world.spawn_actor(

    lidar_bp,

    lidar_transform,

    attach_to=ego_vehicle

)

# ============================================================

# CAMERA CALLBACK

# ============================================================

def camera_callback(image):

    global latest_rgb

    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    array = array.reshape(
        image.height,
        image.width,
        4
    )

    rgb = array[:, :, :3].copy()
    latest_rgb = rgb

    rgb_frames[int(image.frame)] = {
        "image": rgb,
        "transform": image.transform,
        "timestamp": float(image.timestamp)
    }

    if len(rgb_frames) > SENSOR_BUFFER_SIZE:
        oldest = min(rgb_frames)
        del rgb_frames[oldest]


# ============================================================
# LIDAR CALLBACK
# ============================================================

def lidar_callback(point_cloud):

    global latest_lidar
    global latest_lidar_transform
    global latest_lidar_timestamp

    points = np.frombuffer(
        point_cloud.raw_data,
        dtype=np.float32
    )

    points = points.reshape(-1, 4).copy()
    transform = point_cloud.transform

    latest_lidar = points
    latest_lidar_transform = transform
    latest_lidar_timestamp = float(point_cloud.timestamp)

    lidar_frames[int(point_cloud.frame)] = {
        "points": points,
        "transform": transform,
        "timestamp": float(point_cloud.timestamp)
    }

    if len(lidar_frames) > SENSOR_BUFFER_SIZE:
        oldest = min(lidar_frames)
        del lidar_frames[oldest]


# ============================================================
# SYNCHRONIZED SENSOR FRAME
# ============================================================

def get_synchronized_sensor_frame():

    global last_processed_frame

    if not rgb_frames or not lidar_frames:
        return None

    common_frames = set(rgb_frames).intersection(lidar_frames)
    common_frames = [
        frame_id
        for frame_id in common_frames
        if frame_id > last_processed_frame
    ]

    if not common_frames:
        return None

    # Newest common frame keeps latency low while guaranteeing
    # RGB and LiDAR belong to the same CARLA simulation frame.
    frame_id = max(common_frames)

    rgb_data = rgb_frames[frame_id]
    lidar_data = lidar_frames[frame_id]
    last_processed_frame = frame_id

    for old_frame in list(rgb_frames):
        if old_frame <= frame_id:
            del rgb_frames[old_frame]

    for old_frame in list(lidar_frames):
        if old_frame <= frame_id:
            del lidar_frames[old_frame]

    return {
        "frame_id": frame_id,
        "rgb": rgb_data["image"],
        "camera_transform": rgb_data["transform"],
        "lidar": lidar_data["points"],
        "lidar_transform": lidar_data["transform"],
        "timestamp": rgb_data["timestamp"]
    }


# ============================================================

# START SENSORS

# ============================================================

camera.listen(

    camera_callback

)

lidar.listen(

    lidar_callback

)

print("Camera started.")

print("LiDAR started.")

print("Risk engine started.")

# ============================================================

# CAMERA INTRINSICS

# ============================================================

focal_length = WIDTH / (

    2.0 *

    math.tan(

        math.radians(FOV) / 2.0

    )

)

cx = WIDTH / 2.0

cy = HEIGHT / 2.0

K = np.array(

    [

        [focal_length, 0, cx],

        [0, focal_length, cy],

        [0, 0, 1]

    ],

    dtype=np.float32

)

# ============================================================

# LIDAR → CAMERA

# ============================================================

def lidar_to_camera(
    points,
    lidar_transform,
    camera_transform
):

    if points is None:

        return []

    if lidar_transform is None:

        return []

    if camera_transform is None:

        return []

    camera_inverse = np.array(

        camera_transform.get_inverse_matrix()

    )

    projected = []

    for point in points:

        x, y, z = point[:3]

        if not np.isfinite(x):

            continue

        if not np.isfinite(y):

            continue

        if not np.isfinite(z):

            continue

        location = (

            lidar_transform.transform(

                carla.Location(

                    x=float(x),

                    y=float(y),

                    z=float(z)

                )

            )

        )

        world_point = np.array(

            [

                location.x,

                location.y,

                location.z,

                1.0

            ],

            dtype=np.float32

        )

        camera_point = (

            camera_inverse @ world_point

        )[:3]

        if not np.all(

            np.isfinite(camera_point)

        ):

            continue

        # CARLA camera:

        # X = forward

        # Y = right

        # Z = up

        #

        # OpenCV:

        # X = right

        # Y = down

        # Z = forward

        x_cv = camera_point[1]

        y_cv = -camera_point[2]

        z_cv = camera_point[0]

        if z_cv <= 0.1:

            continue

        pixel = K @ np.array(

            [

                x_cv,

                y_cv,

                z_cv

            ],

            dtype=np.float32

        )

        if abs(pixel[2]) < 1e-6:

            continue

        u = pixel[0] / pixel[2]

        v = pixel[1] / pixel[2]

        if not np.isfinite(u):

            continue

        if not np.isfinite(v):

            continue

        u = int(u)

        v = int(v)

        if (

            0 <= u < WIDTH

            and

            0 <= v < HEIGHT

        ):

            projected.append(

                (

                    u,

                    v,

                    x,

                    y,

                    z

                )

            )

    return projected

# ============================================================

# GET OBJECT LIDAR DATA

# ============================================================

def get_lidar_object_data(

    bbox,

    projected_points

):

    x1, y1, x2, y2 = bbox

    points = []

    for point in projected_points:

        u, v, x, y, z = point

        if (

            x1 <= u <= x2

            and

            y1 <= v <= y2

        ):

            points.append(

                (

                    x,

                    y,

                    z

                )

            )

    if len(points) < 3:

        return None

    points = np.array(

        points,

        dtype=np.float32

    )

    # --------------------------------------------------------

    # Remove LiDAR outliers

    # --------------------------------------------------------

    median = np.median(

        points,

        axis=0

    )

    distances = np.linalg.norm(

        points - median,

        axis=1

    )

    points = points[

        distances < 3.0

    ]

    if len(points) < 3:

        return None

    # --------------------------------------------------------

    # Object center

    # --------------------------------------------------------

    x = float(

        np.median(points[:, 0])

    )

    y = float(

        np.median(points[:, 1])

    )

    z = float(

        np.median(points[:, 2])

    )

    # --------------------------------------------------------

    # Object lateral extent

    # --------------------------------------------------------

    y_min = float(

        np.percentile(

            points[:, 1],

            10

        )

    )

    y_max = float(

        np.percentile(

            points[:, 1],

            90

        )

    )

    object_half_width = (

        y_max - y_min

    ) / 2.0

    # Avoid unrealistically tiny widths.

    object_half_width = max(

        object_half_width,

        0.25

    )

    # --------------------------------------------------------

    # Distance

    # --------------------------------------------------------

    distance = math.sqrt(

        x * x +

        y * y +

        z * z

    )

    return {

        "x": x,

        "y": y,

        "z": z,

        "distance": distance,

        "y_min": y_min,

        "y_max": y_max,

        "object_half_width":

            object_half_width,

        "points": len(points)

    }

# ============================================================

# POSITION SMOOTHING

# ============================================================

def smooth_position(

    key,

    x,

    y,

    z

):

    previous = filtered_positions.get(

        key

    )

    if previous is None:

        filtered_positions[key] = (

            x,

            y,

            z

        )

        return x, y, z

    px, py, pz = previous

    sx = (

        POSITION_ALPHA * x

        +

        (1.0 - POSITION_ALPHA) * px

    )

    sy = (

        POSITION_ALPHA * y

        +

        (1.0 - POSITION_ALPHA) * py

    )

    sz = (

        POSITION_ALPHA * z

        +

        (1.0 - POSITION_ALPHA) * pz

    )

    filtered_positions[key] = (

        sx,

        sy,

        sz

    )

    return sx, sy, sz

# ============================================================

# DISTANCE SMOOTHING

# ============================================================

def smooth_distance(

    key,

    distance

):

    previous = filtered_distances.get(

        key

    )

    if previous is None:

        filtered_distances[key] = (

            distance

        )

        return distance

    value = (

        DISTANCE_ALPHA * distance

        +

        (1.0 - DISTANCE_ALPHA) * previous

    )

    filtered_distances[key] = value

    return value

# ============================================================

# VELOCITY

# ============================================================

def calculate_velocity(

    key,

    x,

    timestamp

):

    if key not in track_history:

        track_history[key] = deque(

            maxlen=VELOCITY_HISTORY_SIZE

        )

    history = track_history[key]

    history.append(

        (

            timestamp,

            x

        )

    )

    if len(history) < 3:

        return 0.0

    t1, x1 = history[0]

    t2, x2 = history[-1]

    dt = t2 - t1

    if dt <= 0.05:

        return 0.0

    # Positive = object getting closer

    closing_speed = (

        x1 - x2

    ) / dt

    closing_speed = float(

        np.clip(

            closing_speed,

            -30.0,

            30.0

        )

    )

    if abs(closing_speed) < 0.3:

        closing_speed = 0.0

    return closing_speed

# ============================================================

# TTC

# ============================================================

def calculate_ttc(

    distance,

    closing_speed

):

    if closing_speed <= 0.3:

        return float("inf")

    if distance <= 0:

        return 0.0

    return distance / closing_speed

# ============================================================

# COLLISION CORRIDOR

# ============================================================

def is_in_collision_corridor(

    object_y_min,

    object_y_max,

    object_half_width

):

    # --------------------------------------------------------

    # Ego vehicle occupies:

    #

    # -EGO_HALF_WIDTH → +EGO_HALF_WIDTH

    # --------------------------------------------------------

    ego_left = (

        -EGO_HALF_WIDTH

        -

        SAFETY_MARGIN

    )

    ego_right = (

        EGO_HALF_WIDTH

        +

        SAFETY_MARGIN

    )

    # --------------------------------------------------------

    # Object occupies:

    #

    # object_y_min → object_y_max

    # --------------------------------------------------------

    object_left = object_y_min

    object_right = object_y_max

    # --------------------------------------------------------

    # Check interval overlap

    # --------------------------------------------------------

    overlap = (

        object_right >= ego_left

        and

        object_left <= ego_right

    )

    return overlap

# ============================================================

# RISK ENGINE

# ============================================================

def calculate_risk(

    distance,

    ttc,

    closing_speed,

    in_corridor

):

    # --------------------------------------------------------

    # NOT IN PATH

    # --------------------------------------------------------

    if not in_corridor:

        return "SAFE"

    # --------------------------------------------------------

    # VERY CLOSE

    # --------------------------------------------------------

    if distance <= CRITICAL_DISTANCE:

        return "CRITICAL"

    # --------------------------------------------------------

    # TTC

    # --------------------------------------------------------

    if ttc < TTC_CRITICAL:

        return "CRITICAL"

    if ttc < TTC_WARNING:

        return "WARNING"

    if ttc < TTC_CAUTION:

        return "CAUTION"

    # --------------------------------------------------------

    # DISTANCE FALLBACK

    # --------------------------------------------------------

    if distance <= WARNING_DISTANCE:

        return "WARNING"

    if distance <= CAUTION_DISTANCE:

        return "CAUTION"

    return "SAFE"

# ============================================================

# COLORS

# ============================================================

def risk_color(risk):

    if risk == "CRITICAL":

        return (0, 0, 255)

    if risk == "WARNING":

        return (0, 140, 255)

    if risk == "CAUTION":

        return (0, 220, 255)

    return (0, 220, 0)

# ============================================================

# MAIN LOOP

# ============================================================

try:

    while True:

        synchronized = get_synchronized_sensor_frame()

        if synchronized is None:
            time.sleep(SYNC_WAIT_SLEEP)
            continue

        frame_id = synchronized["frame_id"]
        frame = synchronized["rgb"].copy()
        lidar_points = synchronized["lidar"]
        camera_transform_frame = synchronized["camera_transform"]
        lidar_transform_frame = synchronized["lidar_transform"]
        timestamp = synchronized["timestamp"]

        # YOLO + BYTE TRACK

        # ----------------------------------------------------

        results = model.track(

            frame,

            persist=True,

            tracker="bytetrack.yaml",

            verbose=False

        )

        if not results:

            continue

        result = results[0]

        boxes = result.boxes

        projected_points = (

            lidar_to_camera(

                synchronized["lidar"],

                lidar_transform_frame,

                camera_transform_frame

            )

        )

        overall_risk = "SAFE"
        nearest_distance = float("inf")
        nearest_ttc = float("inf")
        nearest_class = "NONE"
        nearest_id = -1

        priority = {

            "SAFE": 0,

            "CAUTION": 1,

            "WARNING": 2,

            "CRITICAL": 3

        }

        # ----------------------------------------------------

        # PROCESS OBJECTS

        # ----------------------------------------------------

        if boxes is not None:

            for box in boxes:

                if box.id is None:

                    continue

                track_id = int(

                    box.id[0]

                )

                class_id = int(

                    box.cls[0]

                )

                confidence = float(

                    box.conf[0]

                )

                class_name = (

                    model.names[class_id]

                )

                allowed_classes = {

                    "car",

                    "truck",

                    "bus",

                    "motorcycle",

                    "bicycle",

                    "person"

                }

                if class_name not in (

                    allowed_classes

                ):

                    continue

                x1, y1, x2, y2 = map(

                    int,

                    box.xyxy[0]

                )

                key = (

                    class_name,

                    track_id

                )

                # ------------------------------------------------

                # LIDAR DATA

                # ------------------------------------------------

                lidar_data = (

                    get_lidar_object_data(

                        (

                            x1,

                            y1,

                            x2,

                            y2

                        ),

                        projected_points

                    )

                )

                if lidar_data is None:

                    continue

                x = lidar_data["x"]

                y = lidar_data["y"]

                z = lidar_data["z"]

                distance = (

                    lidar_data["distance"]

                )

                y_min = (

                    lidar_data["y_min"]

                )

                y_max = (

                    lidar_data["y_max"]

                )

                object_half_width = (

                    lidar_data[

                        "object_half_width"

                    ]

                )

                # ------------------------------------------------

                # SMOOTH POSITION

                # ------------------------------------------------

                smooth_x, smooth_y, smooth_z = (

                    smooth_position(

                        key,

                        x,

                        y,

                        z

                    )

                )

                # ------------------------------------------------

                # SMOOTH DISTANCE

                # ------------------------------------------------

                smooth_dist = (

                    smooth_distance(

                        key,

                        distance

                    )

                )

                # ------------------------------------------------

                # VELOCITY

                # ------------------------------------------------

                closing_speed = (

                    calculate_velocity(

                        key,

                        smooth_x,

                        timestamp

                    )

                )

                # ------------------------------------------------

                # TTC

                # ------------------------------------------------

                ttc = calculate_ttc(

                    smooth_dist,

                    closing_speed

                )

                # ------------------------------------------------

                # COLLISION CORRIDOR

                # ------------------------------------------------

                in_corridor = (

                    is_in_collision_corridor(

                        y_min,

                        y_max,

                        object_half_width

                    )

                )

                # ------------------------------------------------

                # RISK

                # ------------------------------------------------

                risk = calculate_risk(

                    smooth_dist,

                    ttc,

                    closing_speed,

                    in_corridor

                )

                # Track the nearest object that is actually
                # inside the collision corridor.
                if (
                    in_corridor
                    and smooth_dist < nearest_distance
                ):
                    nearest_distance = smooth_dist
                    nearest_ttc = ttc
                    nearest_class = class_name
                    nearest_id = track_id

                if priority[risk] > priority[

                    overall_risk

                ]:

                    overall_risk = risk

                color = risk_color(

                    risk

                )

                # ------------------------------------------------

                # DRAW BOUNDING BOX

                # ------------------------------------------------

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    color,

                    2

                )

                # ------------------------------------------------

                # TEXT

                # ------------------------------------------------

                if math.isinf(ttc):

                    ttc_text = "INF"

                else:

                    ttc_text = (

                        f"{ttc:.1f}s"

                    )

                corridor_text = (

                    "PATH"

                    if in_corridor

                    else "SIDE"

                )

                text_1 = (

                    f"ID:{track_id} "

                    f"{class_name}"

                )

                text_2 = (

                    f"D:{smooth_dist:.1f}m "

                    f"V:{closing_speed:.1f}"

                )

                text_3 = (

                    f"TTC:{ttc_text} "

                    f"{risk}"

                )

                text_4 = (

                    f"{corridor_text} "

                    f"Y:{smooth_y:.1f}"

                )

                cv2.putText(

                    frame,

                    text_1,

                    (

                        x1,

                        max(15, y1 - 58)

                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.43,

                    color,

                    2

                )

                cv2.putText(

                    frame,

                    text_2,

                    (

                        x1,

                        max(30, y1 - 42)

                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.43,

                    color,

                    2

                )

                cv2.putText(

                    frame,

                    text_3,

                    (

                        x1,

                        max(45, y1 - 26)

                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.43,

                    color,

                    2

                )

                cv2.putText(

                    frame,

                    text_4,

                    (

                        x1,

                        max(60, y1 - 10)

                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.43,

                    color,

                    2

                )

        # ====================================================

        publish_environment_state(
            overall_risk, nearest_distance, nearest_ttc,
            nearest_class, nearest_id
        )

        # DRAW COLLISION CORRIDOR

        # ====================================================

        corridor_width = int(

            (

                EGO_HALF_WIDTH

                +

                SAFETY_MARGIN

            )

            /

            10.0

            *

            WIDTH

        )

        corridor_center = WIDTH // 2

        corridor_left = (

            corridor_center

            -

            corridor_width

        )

        corridor_right = (

            corridor_center

            +

            corridor_width

        )

        # Only draw subtle corridor lines

        cv2.line(

            frame,

            (

                corridor_left,

                HEIGHT

            ),

            (

                corridor_center - 25,

                45

            ),

            (100, 100, 100),

            1

        )

        cv2.line(

            frame,

            (

                corridor_right,

                HEIGHT

            ),

            (

                corridor_center + 25,

                45

            ),

            (100, 100, 100),

            1

        )

        # ====================================================

        # TOP HUD

        # ====================================================

        color = risk_color(

            overall_risk

        )

        cv2.rectangle(

            frame,

            (0, 0),

            (WIDTH, 45),

            (20, 20, 20),

            -1

        )

        cv2.putText(

            frame,

            f"ADAS RISK: {overall_risk}",

            (10, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            color,

            2

        )

        # ====================================================

        # LARGE WARNING

        # ====================================================

        if overall_risk == "CRITICAL":

            cv2.rectangle(

                frame,

                (90, 125),

                (550, 210),

                (0, 0, 255),

                -1

            )

            cv2.putText(

                frame,

                "!! COLLISION WARNING !!",

                (115, 178),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.78,

                (255, 255, 255),

                2

            )

        elif overall_risk == "WARNING":

            cv2.rectangle(

                frame,

                (185, 140),

                (455, 200),

                (0, 140, 255),

                -1

            )

            cv2.putText(

                frame,

                "WARNING",

                (260, 180),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (255, 255, 255),

                2

            )

        elif overall_risk == "CAUTION":

            cv2.putText(

                frame,

                "CAUTION",

                (260, 180),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 220, 255),

                2

            )

        # ====================================================

        # DISPLAY

        # ====================================================

        cv2.imshow(

            "ADAS Risk Engine",

            frame

        )

        key = cv2.waitKey(1)

        if key == ord("q"):

            break

finally:

    print(

        "Stopping sensors..."

    )

    camera.stop()

    lidar.stop()

    camera.destroy()

    lidar.destroy()

    cv2.destroyAllWindows()

    print(

        "Risk engine stopped."

    )