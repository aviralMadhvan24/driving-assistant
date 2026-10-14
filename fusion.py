import carla
import cv2
import numpy as np
from ultralytics import YOLO
import queue

# ============================================================
# CONFIG
# ============================================================

HOST = "127.0.0.1"
PORT = 2000

WIDTH = 640
HEIGHT = 360
FOV = 90

CONFIDENCE_THRESHOLD = 0.40

# ============================================================
# YOLO MODEL
# ============================================================

model = YOLO("yolo11n.pt")

# ============================================================
# CONNECT TO CARLA
# ============================================================

client = carla.Client(HOST, PORT)
client.set_timeout(10.0)

world = client.get_world()

# ============================================================
# FIND EGO VEHICLE
# ============================================================

vehicle = None

for v in world.get_actors().filter("vehicle.*"):
    if v.attributes.get("role_name") == "hero":
        vehicle = v
        break

if vehicle is None:
    print("Hero vehicle not found.")
    print("Make sure manual_control.py is running.")
    exit()

print("Using:", vehicle.type_id)
print("Vehicle ID:", vehicle.id)

# ============================================================
# BLUEPRINT LIBRARY
# ============================================================

blueprints = world.get_blueprint_library()

# ============================================================
# CAMERA TRANSFORM
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

# ============================================================
# RGB CAMERA
# ============================================================

rgb_bp = blueprints.find("sensor.camera.rgb")

rgb_bp.set_attribute(
    "image_size_x",
    str(WIDTH)
)

rgb_bp.set_attribute(
    "image_size_y",
    str(HEIGHT)
)

rgb_bp.set_attribute(
    "fov",
    str(FOV)
)

rgb_camera = world.spawn_actor(
    rgb_bp,
    camera_transform,
    attach_to=vehicle
)

# ============================================================
# DEPTH CAMERA
# ============================================================

depth_bp = blueprints.find("sensor.camera.depth")

depth_bp.set_attribute(
    "image_size_x",
    str(WIDTH)
)

depth_bp.set_attribute(
    "image_size_y",
    str(HEIGHT)
)

depth_bp.set_attribute(
    "fov",
    str(FOV)
)

depth_camera = world.spawn_actor(
    depth_bp,
    camera_transform,
    attach_to=vehicle
)

# ============================================================
# QUEUES
# ============================================================

rgb_queue = queue.Queue()
depth_queue = queue.Queue()

rgb_camera.listen(rgb_queue.put)
depth_camera.listen(depth_queue.put)

# ============================================================
# DEPTH CONVERSION
# ============================================================

def depth_to_meters(image):

    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    array = np.reshape(
        array,
        (image.height, image.width, 4)
    )

    # CARLA depth encoding
    depth = (
        array[:, :, 2].astype(np.float32)
        + array[:, :, 1].astype(np.float32) * 256.0
        + array[:, :, 0].astype(np.float32) * 256.0 * 256.0
    )

    depth /= (256.0 ** 3 - 1.0)

    # CARLA depth range = 0 to 1000 meters
    depth_meters = depth * 1000.0

    return depth_meters


# ============================================================
# ROBUST DISTANCE ESTIMATION
# ============================================================

def get_object_distance(depth, x1, y1, x2, y2):

    box_width = x2 - x1
    box_height = y2 - y1

    # Take central portion of bounding box.
    #
    # We avoid the outer edges because they can contain
    # background pixels or another object.

    rx1 = x1 + int(box_width * 0.25)
    rx2 = x1 + int(box_width * 0.75)

    ry1 = y1 + int(box_height * 0.30)
    ry2 = y1 + int(box_height * 0.75)

    # Clamp coordinates

    rx1 = max(0, min(WIDTH, rx1))
    rx2 = max(0, min(WIDTH, rx2))

    ry1 = max(0, min(HEIGHT, ry1))
    ry2 = max(0, min(HEIGHT, ry2))

    # Make sure region is valid

    if rx2 <= rx1 or ry2 <= ry1:
        return -1.0

    depth_region = depth[
        ry1:ry2,
        rx1:rx2
    ]

    # Remove invalid depth values

    valid_depth = depth_region[
        (depth_region > 0.5) &
        (depth_region < 100.0)
    ]

    if len(valid_depth) == 0:
        return -1.0

    # Median is much more robust than a single pixel

    distance = float(
        np.median(valid_depth)
    )

    return distance


# ============================================================
# START
# ============================================================

print()
print("==============================")
print(" RGB + DEPTH FUSION RUNNING")
print("==============================")
print("YOLO:", "yolo11n.pt")
print("Resolution:", f"{WIDTH}x{HEIGHT}")
print("Confidence:", CONFIDENCE_THRESHOLD)
print()
print("Press Q to stop.")
print()

# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # Get synchronized-ish RGB and depth frames
        # ----------------------------------------------------

        rgb_image = rgb_queue.get()

        depth_image = depth_queue.get()

        # ----------------------------------------------------
        # Convert RGB
        # ----------------------------------------------------

        array = np.frombuffer(
            rgb_image.raw_data,
            dtype=np.uint8
        )

        frame = np.reshape(
            array,
            (HEIGHT, WIDTH, 4)
        )[:, :, :3].copy()

        # ----------------------------------------------------
        # Convert Depth
        # ----------------------------------------------------

        depth = depth_to_meters(
            depth_image
        )

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model(
            frame,
            verbose=False
        )

        # ----------------------------------------------------
        # Process detections
        # ----------------------------------------------------

        for box in results[0].boxes:

            confidence = float(
                box.conf[0]
            )

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(
                box.cls[0]
            )

            object_name = model.names[
                class_id
            ]

            # Bounding box

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Clamp bounding box

            x1 = max(0, min(WIDTH - 1, x1))
            x2 = max(0, min(WIDTH - 1, x2))

            y1 = max(0, min(HEIGHT - 1, y1))
            y2 = max(0, min(HEIGHT - 1, y2))

            # ------------------------------------------------
            # Distance
            # ------------------------------------------------

            distance = get_object_distance(
                depth,
                x1,
                y1,
                x2,
                y2
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            if distance > 0:

                label = (
                    f"{object_name} "
                    f"{confidence:.2f} | "
                    f"{distance:.1f}m"
                )

            else:

                label = (
                    f"{object_name} "
                    f"{confidence:.2f} | "
                    f"N/A"
                )

            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Draw label background
            # ------------------------------------------------

            text_y = max(
                20,
                y1 - 10
            )

            cv2.putText(
                frame,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "ADAS RGB + Depth Fusion",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

# ============================================================
# CLEANUP
# ============================================================

except KeyboardInterrupt:

    print("\nStopping...")

finally:

    print("\nDestroying sensors...")

    rgb_camera.stop()
    depth_camera.stop()

    rgb_camera.destroy()
    depth_camera.destroy()

    cv2.destroyAllWindows()

    print("Fusion stopped.")