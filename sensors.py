import carla
import cv2
import numpy as np
import os


# ============================================================
# CONFIG
# ============================================================

HOST = "127.0.0.1"
PORT = 2000

WIDTH = 640
HEIGHT = 360
FOV = 90

SAVE_DATA = True

DATA_DIR = "data"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

if SAVE_DATA:
    os.makedirs(os.path.join(DATA_DIR, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "depth"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "semantic"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "lidar"), exist_ok=True)


# ============================================================
# RGB CAMERA
# ============================================================

def process_rgb(image):

    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    array = np.reshape(
        array,
        (image.height, image.width, 4)
    )

    # CARLA uses BGRA
    frame = array[:, :, :3]

    if SAVE_DATA:
        path = os.path.join(
            DATA_DIR,
            "rgb",
            f"{image.frame:06d}.png"
        )

        cv2.imwrite(path, frame)

    cv2.imshow("RGB Camera", frame)
    cv2.waitKey(1)


# ============================================================
# DEPTH CAMERA
# ============================================================

def process_depth(image):

    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    array = np.reshape(
        array,
        (image.height, image.width, 4)
    )

    # CARLA depth encoding:
    # R + G*256 + B*256^2
    normalized = (
        array[:, :, 2].astype(np.float32)
        + array[:, :, 1].astype(np.float32) * 256
        + array[:, :, 0].astype(np.float32) * 256 * 256
    )

    normalized /= (256 ** 3 - 1)

    # CARLA depth is approximately 0-1000 meters
    depth_meters = normalized * 1000.0

    # Save REAL depth values
    if SAVE_DATA:
        path = os.path.join(
            DATA_DIR,
            "depth",
            f"{image.frame:06d}.npy"
        )

        np.save(path, depth_meters)

    # --------------------------------------------------------
    # Visualization
    # Show only 0-50 meters because that's useful for driving
    # --------------------------------------------------------

    depth_clipped = np.clip(
        depth_meters,
        0,
        50
    )

    depth_display = (
        depth_clipped / 50.0 * 255
    ).astype(np.uint8)

    # Apply color map for easier visualization
    depth_colormap = cv2.applyColorMap(
        depth_display,
        cv2.COLORMAP_JET
    )

    cv2.imshow(
        "Depth Camera",
        depth_colormap
    )

    cv2.waitKey(1)


# ============================================================
# SEMANTIC SEGMENTATION
# ============================================================

def process_semantic(image):

    image.convert(
        carla.ColorConverter.CityScapesPalette
    )

    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    array = np.reshape(
        array,
        (image.height, image.width, 4)
    )

    frame = array[:, :, :3]

    if SAVE_DATA:

        path = os.path.join(
            DATA_DIR,
            "semantic",
            f"{image.frame:06d}.png"
        )

        cv2.imwrite(
            path,
            frame
        )

    cv2.imshow(
        "Semantic Segmentation",
        frame
    )

    cv2.waitKey(1)


# ============================================================
# LIDAR
# ============================================================

def process_lidar(point_cloud):

    points = np.frombuffer(
        point_cloud.raw_data,
        dtype=np.float32
    )

    points = np.reshape(
        points,
        (-1, 4)
    )

    # Columns:
    # X
    # Y
    # Z
    # Intensity

    if SAVE_DATA:

        path = os.path.join(
            DATA_DIR,
            "lidar",
            f"{point_cloud.frame:06d}.npy"
        )

        np.save(
            path,
            points
        )

    print(
        f"[LiDAR] "
        f"frame={point_cloud.frame} "
        f"points={len(points)}"
    )


# ============================================================
# COLLISION SENSOR
# ============================================================

def process_collision(event):

    other_actor = event.other_actor

    print("\n================================")
    print("        !!! COLLISION !!!")
    print("================================")

    print(
        f"Frame       : {event.frame}"
    )

    print(
        f"Other actor : {other_actor.type_id}"
    )

    print("================================\n")


# ============================================================
# LANE INVASION SENSOR
# ============================================================

def process_lane_invasion(event):

    markings = [
        str(marking.type)
        for marking in event.crossed_lane_markings
    ]

    print(
        f"[LANE INVASION] "
        f"frame={event.frame} "
        f"markings={markings}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    client = carla.Client(
        HOST,
        PORT
    )

    client.set_timeout(10.0)

    world = client.get_world()

    print("\n================================")
    print("Connected to CARLA")
    print("================================")

    print(
        "Map:",
        world.get_map().name
    )


    # ========================================================
    # FIND VEHICLE
    # ========================================================

    vehicles = world.get_actors().filter(
        "vehicle.*"
    )

    if len(vehicles) == 0:

        print("\nNo vehicles found.")
        print(
            "Start manual_control.py first."
        )

        return


    # Prefer hero vehicle
    vehicle = None

    for v in vehicles:

        if v.attributes.get(
            "role_name"
        ) == "hero":

            vehicle = v
            break


    # If hero isn't found
    # use first vehicle

    if vehicle is None:

        vehicle = vehicles[0]


    print(
        "\nUsing vehicle:",
        vehicle.type_id
    )

    print(
        "Vehicle ID:",
        vehicle.id
    )


    # ========================================================
    # BLUEPRINT LIBRARY
    # ========================================================

    blueprint_library = (
        world.get_blueprint_library()
    )


    sensors = []


    # ========================================================
    # CAMERA TRANSFORM
    # ========================================================

    # Forward-facing camera
    #
    # x = forward
    # z = height
    #
    # This is intentionally above the hood
    # so we don't see the vehicle body.

    camera_transform = carla.Transform(

        carla.Location(
            x=1.5,
            y=0.0,
            z=2.0
        ),

        carla.Rotation(
            pitch=-5.0,
            yaw=0.0,
            roll=0.0
        )
    )


    # ========================================================
    # RGB CAMERA
    # ========================================================

    rgb_bp = blueprint_library.find(
        "sensor.camera.rgb"
    )

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

    rgb_camera.listen(
        process_rgb
    )

    sensors.append(
        rgb_camera
    )


    # ========================================================
    # DEPTH CAMERA
    # ========================================================

    depth_bp = blueprint_library.find(
        "sensor.camera.depth"
    )

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

    depth_camera.listen(
        process_depth
    )

    sensors.append(
        depth_camera
    )


    # ========================================================
    # SEMANTIC SEGMENTATION
    # ========================================================

    semantic_bp = blueprint_library.find(
        "sensor.camera.semantic_segmentation"
    )

    semantic_bp.set_attribute(
        "image_size_x",
        str(WIDTH)
    )

    semantic_bp.set_attribute(
        "image_size_y",
        str(HEIGHT)
    )

    semantic_bp.set_attribute(
        "fov",
        str(FOV)
    )

    semantic_camera = world.spawn_actor(
        semantic_bp,
        camera_transform,
        attach_to=vehicle
    )

    semantic_camera.listen(
        process_semantic
    )

    sensors.append(
        semantic_camera
    )


    # ========================================================
    # LIDAR
    # ========================================================

    lidar_bp = blueprint_library.find(
        "sensor.lidar.ray_cast"
    )

    lidar_bp.set_attribute(
        "channels",
        "16"
    )

    lidar_bp.set_attribute(
        "range",
        "50"
    )

    lidar_bp.set_attribute(
        "points_per_second",
        "30000"
    )

    lidar_bp.set_attribute(
        "rotation_frequency",
        "10"
    )

    lidar_bp.set_attribute(
        "upper_fov",
        "10"
    )

    lidar_bp.set_attribute(
        "lower_fov",
        "-30"
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
        attach_to=vehicle
    )

    lidar.listen(
        process_lidar
    )

    sensors.append(
        lidar
    )


    # ========================================================
    # COLLISION SENSOR
    # ========================================================

    collision_bp = blueprint_library.find(
        "sensor.other.collision"
    )

    collision_sensor = world.spawn_actor(
        collision_bp,
        carla.Transform(),
        attach_to=vehicle
    )

    collision_sensor.listen(
        process_collision
    )

    sensors.append(
        collision_sensor
    )


    # ========================================================
    # LANE INVASION SENSOR
    # ========================================================

    lane_bp = blueprint_library.find(
        "sensor.other.lane_invasion"
    )

    lane_sensor = world.spawn_actor(
        lane_bp,
        carla.Transform(),
        attach_to=vehicle
    )

    lane_sensor.listen(
        process_lane_invasion
    )

    sensors.append(
        lane_sensor
    )


    # ========================================================
    # READY
    # ========================================================

    print("\n======================================")
    print("        ALL SENSORS RUNNING")
    print("======================================")

    print(
        "RGB          : 640x360"
    )

    print(
        "Depth        : 640x360"
    )

    print(
        "Semantic     : 640x360"
    )

    print(
        "LiDAR        : 16 channels"
    )

    print(
        "LiDAR range  : 50m"
    )

    print(
        "LiDAR points : 30,000/sec"
    )

    print(
        "\nDrive the vehicle using manual_control.py"
    )

    print(
        "Press Q in any OpenCV window to stop."
    )


    # ========================================================
    # KEEP PROGRAM RUNNING
    # ========================================================

    try:

        while True:

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break


    except KeyboardInterrupt:

        print(
            "\nStopping sensors..."
        )


    finally:

        print(
            "\nDestroying sensors..."
        )

        for sensor in sensors:

            try:

                sensor.stop()
                sensor.destroy()

            except Exception:
                pass


        cv2.destroyAllWindows()

        print(
            "Sensors stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()