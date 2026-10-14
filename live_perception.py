import carla
import cv2
import numpy as np
from ultralytics import YOLO

HOST = "127.0.0.1"
PORT = 2000

WIDTH = 640
HEIGHT = 360
FOV = 90

model = YOLO("yolo11n.pt")

client = carla.Client(HOST, PORT)
client.set_timeout(10.0)

world = client.get_world()

vehicles = world.get_actors().filter("vehicle.*")

vehicle = None

for v in vehicles:
    if v.attributes.get("role_name") == "hero":
        vehicle = v
        break

if vehicle is None:
    print("Hero vehicle not found.")
    exit()

print("Using:", vehicle.type_id)
print("Vehicle ID:", vehicle.id)


def process_image(image):

    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    array = np.reshape(
        array,
        (image.height, image.width, 4)
    )

    frame = array[:, :, :3]

    results = model(
        frame,
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow(
        "Live YOLO Perception",
        annotated
    )

    cv2.waitKey(1)


blueprint_library = world.get_blueprint_library()

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
    attach_to=vehicle
)

camera.listen(process_image)

print("\n==============================")
print(" LIVE YOLO PERCEPTION RUNNING")
print("==============================")
print("Press Q to stop.")

try:

    while True:

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

except KeyboardInterrupt:
    pass

finally:

    camera.stop()
    camera.destroy()

    cv2.destroyAllWindows()

    print("Camera stopped.")