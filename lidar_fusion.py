import carla
import cv2
import numpy as np
import math
import time
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

WIDTH = 640
HEIGHT = 360
FOV = 90

LIDAR_RANGE = 50.0

# Distance smoothing
DISTANCE_ALPHA = 0.10

# Forward-position smoothing
POSITION_ALPHA = 0.30

# Velocity smoothing
VELOCITY_ALPHA = 0.25

# Ignore tiny velocity fluctuations
VELOCITY_DEADBAND = 0.30

# Minimum closing speed for TTC
MIN_CLOSING_SPEED = 0.50

# TTC thresholds
TTC_CRITICAL = 1.5
TTC_WARNING = 3.0
TTC_CAUTION = 6.0

# Remove tracks after this much simulation time
TRACK_TIMEOUT = 2.0


# ============================================================
# YOLO
# ============================================================

model = YOLO("yolo11n.pt")


# ============================================================
# CARLA CONNECTION
# ============================================================

client = carla.Client(
    "127.0.0.1",
    2000
)

client.set_timeout(10.0)

world = client.get_world()


# ============================================================
# FIND TESLA MODEL 3
# ============================================================

vehicle = None

for actor in world.get_actors().filter("vehicle.*"):

    if "model3" in actor.type_id:

        vehicle = actor
        break


if vehicle is None:

    print("Tesla Model 3 not found.")
    print("Start manual_control.py first.")
    exit()


print(
    "Using vehicle:",
    vehicle.type_id
)


# ============================================================
# RGB CAMERA
# ============================================================

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


camera_bp = world.get_blueprint_library().find(
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


camera = world.spawn_actor(

    camera_bp,

    camera_transform,

    attach_to=vehicle

)


# ============================================================
# LiDAR
# ============================================================

lidar_transform = carla.Transform(

    carla.Location(
        x=0.0,
        y=0.0,
        z=2.2
    ),

    carla.Rotation(
        pitch=0.0,
        yaw=0.0,
        roll=0.0
    )
)


lidar_bp = world.get_blueprint_library().find(
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


lidar = world.spawn_actor(

    lidar_bp,

    lidar_transform,

    attach_to=vehicle

)


# ============================================================
# SENSOR DATA
# ============================================================

latest_rgb = None

latest_lidar = None

latest_lidar_transform = None

latest_lidar_timestamp = None


# ============================================================
# TRACKING / SMOOTHING STORAGE
# ============================================================

filtered_distances = {}

filtered_positions = {}

track_history = {}


# ============================================================
# DISTANCE SMOOTHING
# ============================================================

def smooth_distance(
    object_key,
    current_distance
):

    if object_key not in filtered_distances:

        filtered_distances[
            object_key
        ] = current_distance

    else:

        previous = filtered_distances[
            object_key
        ]

        filtered_distances[
            object_key
        ] = (

            DISTANCE_ALPHA
            * current_distance

            +

            (1.0 - DISTANCE_ALPHA)
            * previous

        )

    return filtered_distances[
        object_key
    ]


# ============================================================
# POSITION SMOOTHING
# ============================================================

def smooth_position(
    object_key,
    current_x
):

    if object_key not in filtered_positions:

        filtered_positions[
            object_key
        ] = current_x

    else:

        previous = filtered_positions[
            object_key
        ]

        filtered_positions[
            object_key
        ] = (

            POSITION_ALPHA
            * current_x

            +

            (1.0 - POSITION_ALPHA)
            * previous

        )

    return filtered_positions[
        object_key
    ]


# ============================================================
# RGB CALLBACK
# ============================================================

def camera_callback(image):

    global latest_rgb

    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    frame = np.reshape(

        array,

        (
            HEIGHT,
            WIDTH,
            4
        )

    )[:, :, :3].copy()


    frame = cv2.cvtColor(

        frame,

        cv2.COLOR_RGB2BGR

    )


    latest_rgb = frame


# ============================================================
# LiDAR CALLBACK
# ============================================================

def lidar_callback(point_cloud):

    global latest_lidar
    global latest_lidar_transform
    global latest_lidar_timestamp


    points = np.frombuffer(

        point_cloud.raw_data,

        dtype=np.float32

    ).reshape(-1, 4)


    latest_lidar = points


    # Use the transform corresponding to this LiDAR measurement
    latest_lidar_transform = (
        lidar.get_transform()
    )


    # IMPORTANT:
    # CARLA simulation timestamp
    latest_lidar_timestamp = (
        point_cloud.timestamp
    )


# ============================================================
# CAMERA INTRINSICS
# ============================================================

focal_length = WIDTH / (

    2.0
    *
    math.tan(
        math.radians(FOV) / 2.0
    )

)


cx = WIDTH / 2.0
cy = HEIGHT / 2.0


K = np.array(

    [

        [
            focal_length,
            0,
            cx
        ],

        [
            0,
            focal_length,
            cy
        ],

        [
            0,
            0,
            1
        ]

    ],

    dtype=np.float64

)


# ============================================================
# LiDAR → CAMERA PROJECTION
# ============================================================

def lidar_to_camera(point):

    global latest_lidar_transform


    if latest_lidar_transform is None:

        return None


    x, y, z = point[:3]


    # --------------------------------------------------------
    # Validate raw LiDAR point
    # --------------------------------------------------------

    if not (

        np.isfinite(x)
        and
        np.isfinite(y)
        and
        np.isfinite(z)

    ):

        return None


    # --------------------------------------------------------
    # LiDAR local coordinate
    # --------------------------------------------------------

    lidar_location = carla.Location(

        x=float(x),
        y=float(y),
        z=float(z)

    )


    # --------------------------------------------------------
    # LiDAR → World
    # --------------------------------------------------------

    world_location = (

        latest_lidar_transform.transform(

            lidar_location

        )

    )


    # --------------------------------------------------------
    # World → Camera
    # --------------------------------------------------------

    camera_inverse = (

        camera
        .get_transform()
        .get_inverse_matrix()

    )


    world_point = np.array(

        [

            world_location.x,
            world_location.y,
            world_location.z,
            1.0

        ],

        dtype=np.float64

    )


    camera_point = np.dot(

        np.array(

            camera_inverse,

            dtype=np.float64

        ),

        world_point

    )


    x_carla = camera_point[0]

    y_carla = camera_point[1]

    z_carla = camera_point[2]


    # --------------------------------------------------------
    # Validate transformed point
    # --------------------------------------------------------

    if not np.all(

        np.isfinite(

            [
                x_carla,
                y_carla,
                z_carla
            ]

        )

    ):

        return None


    # --------------------------------------------------------
    # CARLA camera → OpenCV camera
    # --------------------------------------------------------

    x_cv = y_carla

    y_cv = -z_carla

    z_cv = x_carla


    # Behind camera
    if z_cv <= 0.1:

        return None


    # --------------------------------------------------------
    # Project to image
    # --------------------------------------------------------

    pixel = np.dot(

        K,

        np.array(

            [
                x_cv,
                y_cv,
                z_cv
            ],

            dtype=np.float64

        )

    )


    # --------------------------------------------------------
    # Validate projection
    # --------------------------------------------------------

    if not np.all(

        np.isfinite(pixel)

    ):

        return None


    if abs(pixel[2]) < 1e-6:

        return None


    projected_x = (

        pixel[0]
        /
        pixel[2]

    )


    projected_y = (

        pixel[1]
        /
        pixel[2]

    )


    if not np.isfinite(
        projected_x
    ):

        return None


    if not np.isfinite(
        projected_y
    ):

        return None


    u = int(projected_x)

    v = int(projected_y)


    # --------------------------------------------------------
    # Image boundaries
    # --------------------------------------------------------

    if (

        u < 0
        or
        u >= WIDTH
        or
        v < 0
        or
        v >= HEIGHT

    ):

        return None


    return (

        u,
        v,

        x_carla,
        y_carla,
        z_carla

    )


# ============================================================
# GET LiDAR DATA FOR OBJECT
# ============================================================

def get_lidar_object_data(

    bbox,
    lidar_points

):

    if lidar_points is None:

        return None


    x1, y1, x2, y2 = bbox


    object_points = []


    # --------------------------------------------------------
    # Project LiDAR points
    # --------------------------------------------------------

    for point in lidar_points:

        projection = lidar_to_camera(
            point
        )


        if projection is None:

            continue


        u, v, _, _, _ = projection


        if (

            x1 <= u <= x2

            and

            y1 <= v <= y2

        ):

            object_points.append(
                point
            )


    # Need enough points
    if len(object_points) < 5:

        return None


    object_points = np.array(
        object_points
    )


    # --------------------------------------------------------
    # Calculate distances
    # --------------------------------------------------------

    distances = np.linalg.norm(

        object_points[:, :3],

        axis=1

    )


    median_distance = np.median(
        distances
    )


    # --------------------------------------------------------
    # Remove outliers
    # --------------------------------------------------------

    mask = (

        np.abs(

            distances
            -
            median_distance

        )

        <

        3.0

    )


    object_points = (
        object_points[mask]
    )


    if len(object_points) < 5:

        return None


    # --------------------------------------------------------
    # Median 3D position
    # --------------------------------------------------------

    position = np.median(

        object_points[:, :3],

        axis=0

    )


    pos_x = float(
        position[0]
    )

    pos_y = float(
        position[1]
    )

    pos_z = float(
        position[2]
    )


    # --------------------------------------------------------
    # Euclidean distance
    # --------------------------------------------------------

    distance = math.sqrt(

        pos_x ** 2
        +
        pos_y ** 2
        +
        pos_z ** 2

    )


    return {

        "distance": distance,

        "x": pos_x,

        "y": pos_y,

        "z": pos_z,

        "points": len(
            object_points
        )

    }


# ============================================================
# RELATIVE VELOCITY
# ============================================================

def calculate_relative_velocity(

    object_key,

    current_x,

    current_timestamp

):

    previous = track_history.get(
        object_key
    )


    # --------------------------------------------------------
    # First observation
    # --------------------------------------------------------

    if previous is None:

        track_history[
            object_key
        ] = {

            "x": current_x,

            "time": current_timestamp,

            "velocity": 0.0

        }

        return 0.0


    previous_x = previous[
        "x"
    ]

    previous_time = previous[
        "time"
    ]


    # --------------------------------------------------------
    # CARLA simulation time difference
    # --------------------------------------------------------

    dt = (

        current_timestamp
        -
        previous_time

    )


    if dt <= 0.001:

        return previous[
            "velocity"
        ]


    # --------------------------------------------------------
    # Relative velocity
    #
    # X decreases → object approaching
    # X increases → object moving away
    # --------------------------------------------------------

    raw_velocity = (

        previous_x
        -
        current_x

    ) / dt


    # --------------------------------------------------------
    # Reject unrealistic spikes
    # --------------------------------------------------------

    raw_velocity = float(

        np.clip(

            raw_velocity,

            -30.0,
            30.0

        )

    )


    previous_velocity = (
        previous["velocity"]
    )


    # --------------------------------------------------------
    # Velocity EMA
    # --------------------------------------------------------

    velocity = (

        VELOCITY_ALPHA
        *
        raw_velocity

        +

        (1.0 - VELOCITY_ALPHA)
        *
        previous_velocity

    )


    # --------------------------------------------------------
    # Deadband
    # --------------------------------------------------------

    if abs(velocity) < VELOCITY_DEADBAND:

        velocity = 0.0


    # --------------------------------------------------------
    # Store history
    # --------------------------------------------------------

    track_history[
        object_key
    ] = {

        "x": current_x,

        "time": current_timestamp,

        "velocity": velocity

    }


    return velocity


# ============================================================
# TTC
# ============================================================

def calculate_ttc(

    distance,

    closing_speed

):

    # Not approaching
    if (

        closing_speed
        <=
        MIN_CLOSING_SPEED

    ):

        return float("inf")


    if distance <= 0:

        return 0.0


    return (

        distance
        /
        closing_speed

    )


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(ttc):

    if math.isinf(ttc):

        return "SAFE"


    if ttc < TTC_CRITICAL:

        return "CRITICAL"


    elif ttc < TTC_WARNING:

        return "WARNING"


    elif ttc < TTC_CAUTION:

        return "CAUTION"


    else:

        return "SAFE"


# ============================================================
# START SENSOR CALLBACKS
# ============================================================

camera.listen(
    camera_callback
)

lidar.listen(
    lidar_callback
)


print()
print("======================================")
print(" LiDAR + YOLO + TRACKING + TTC")
print("======================================")
print("Timestamp-based velocity enabled")
print("Waiting for sensor data...")
print()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # RGB available?
        # ----------------------------------------------------

        if latest_rgb is None:

            time.sleep(0.01)

            continue


        # ----------------------------------------------------
        # LiDAR available?
        # ----------------------------------------------------

        if latest_lidar is None:

            time.sleep(0.01)

            continue


        # ----------------------------------------------------
        # Timestamp available?
        # ----------------------------------------------------

        if latest_lidar_timestamp is None:

            time.sleep(0.01)

            continue


        frame = latest_rgb.copy()

        lidar_points = latest_lidar

        current_timestamp = (
            latest_lidar_timestamp
        )


        # ====================================================
        # YOLO + BYTE TRACK
        # ====================================================

        results = model.track(

            frame,

            persist=True,

            tracker="bytetrack.yaml",

            verbose=False

        )


        result = results[0]


        # ====================================================
        # TRACKED OBJECTS
        # ====================================================

        if result.boxes is not None:

            boxes = result.boxes


            for i in range(
                len(boxes)
            ):

                box = boxes[i]


                # ------------------------------------------------
                # Track ID
                # ------------------------------------------------

                if box.id is None:

                    continue


                track_id = int(
                    box.id.item()
                )


                confidence = float(
                    box.conf.item()
                )


                class_id = int(
                    box.cls.item()
                )


                class_name = (
                    model.names[
                        class_id
                    ]
                )


                # ------------------------------------------------
                # Relevant classes
                # ------------------------------------------------

                allowed_classes = {

                    "car",
                    "truck",
                    "bus",
                    "motorcycle",
                    "bicycle",
                    "person"

                }


                if (

                    class_name
                    not in
                    allowed_classes

                ):

                    continue


                # ------------------------------------------------
                # Bounding box
                # ------------------------------------------------

                x1, y1, x2, y2 = map(

                    int,

                    box.xyxy[
                        0
                    ].tolist()

                )


                # ------------------------------------------------
                # LiDAR object data
                # ------------------------------------------------

                lidar_data = (

                    get_lidar_object_data(

                        (
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                        lidar_points

                    )

                )


                if lidar_data is None:

                    continue


                distance = (
                    lidar_data[
                        "distance"
                    ]
                )


                pos_x = (
                    lidar_data["x"]
                )

                pos_y = (
                    lidar_data["y"]
                )

                pos_z = (
                    lidar_data["z"]
                )

                point_count = (
                    lidar_data[
                        "points"
                    ]
                )


                # ------------------------------------------------
                # Unique track key
                # ------------------------------------------------

                object_key = (

                    class_name,

                    track_id

                )


                # =================================================
                # DISTANCE SMOOTHING
                # =================================================

                smooth_distance_value = (

                    smooth_distance(

                        object_key,

                        distance

                    )

                )


                # =================================================
                # POSITION SMOOTHING
                # =================================================

                smooth_x = (

                    smooth_position(

                        object_key,

                        pos_x

                    )

                )


                # =================================================
                # RELATIVE VELOCITY
                # =================================================

                velocity = (

                    calculate_relative_velocity(

                        object_key,

                        smooth_x,

                        current_timestamp

                    )

                )


                # =================================================
                # TTC
                # =================================================

                ttc = calculate_ttc(

                    smooth_distance_value,

                    velocity

                )


                # =================================================
                # RISK
                # =================================================

                risk = get_risk_level(
                    ttc
                )


                # =================================================
                # COLOR
                # =================================================

                if risk == "CRITICAL":

                    color = (
                        0,
                        0,
                        255
                    )

                elif risk == "WARNING":

                    color = (
                        0,
                        100,
                        255
                    )

                elif risk == "CAUTION":

                    color = (
                        0,
                        255,
                        255
                    )

                else:

                    color = (
                        0,
                        255,
                        0
                    )


                # =================================================
                # DRAW BOX
                # =================================================

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    color,

                    2

                )


                # =================================================
                # TTC TEXT
                # =================================================

                if math.isinf(ttc):

                    ttc_text = (
                        "TTC: --"
                    )

                else:

                    ttc_text = (

                        f"TTC: {ttc:.1f}s"

                    )


                # =================================================
                # LABELS
                # =================================================

                label1 = (

                    f"ID:{track_id} "
                    f"{class_name} "
                    f"{confidence:.2f}"

                )


                label2 = (

                    f"D:{smooth_distance_value:.1f}m "
                    f"V:{velocity:.1f}m/s"

                )


                label3 = (

                    f"{ttc_text} "
                    f"RISK:{risk}"

                )


                label4 = (

                    f"X:{pos_x:.1f} "
                    f"Y:{pos_y:.1f} "
                    f"Z:{pos_z:.1f}"

                )


                label5 = (

                    f"LiDAR pts:{point_count}"

                )


                # =================================================
                # TEXT POSITION
                # =================================================

                text_y = max(

                    20,

                    y1 - 55

                )


                # =================================================
                # DRAW LABELS
                # =================================================

                cv2.putText(

                    frame,

                    label1,

                    (
                        x1,
                        text_y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    color,

                    1,

                    cv2.LINE_AA

                )


                cv2.putText(

                    frame,

                    label2,

                    (
                        x1,
                        text_y + 17
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    color,

                    1,

                    cv2.LINE_AA

                )


                cv2.putText(

                    frame,

                    label3,

                    (
                        x1,
                        text_y + 34
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    color,

                    1,

                    cv2.LINE_AA

                )


                cv2.putText(

                    frame,

                    label4,

                    (
                        x1,
                        y2 + 18
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.40,

                    color,

                    1,

                    cv2.LINE_AA

                )


                cv2.putText(

                    frame,

                    label5,

                    (
                        x1,
                        y2 + 35
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.40,

                    color,

                    1,

                    cv2.LINE_AA

                )


        # ====================================================
        # CLEAN STALE TRACKS
        # ====================================================

        stale_keys = []


        for key, data in (
            track_history.items()
        ):

            if (

                current_timestamp
                -
                data["time"]

                >

                TRACK_TIMEOUT

            ):

                stale_keys.append(
                    key
                )


        for key in stale_keys:

            if key in track_history:

                del track_history[
                    key
                ]


            if key in filtered_distances:

                del filtered_distances[
                    key
                ]


            if key in filtered_positions:

                del filtered_positions[
                    key
                ]


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(

            "CARLA LiDAR + YOLO + TTC",

            frame

        )


        key = (

            cv2.waitKey(1)
            &
            0xFF

        )


        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

finally:

    print(
        "Cleaning up..."
    )


    camera.stop()

    lidar.stop()


    camera.destroy()

    lidar.destroy()


    cv2.destroyAllWindows()


    print(
        "Done."
    )