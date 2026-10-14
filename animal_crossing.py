
import carla
import random
import time

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(15)

world = client.get_world()
blueprints = world.get_blueprint_library()

# Find the Tesla, waiting briefly if it is still spawning.
ego = None
for _ in range(10):
    vehicles = world.get_actors().filter("vehicle.*")
    ego = next(
        (v for v in vehicles
         if v.attributes.get("role_name") == "hero"),
        None
    )
    if ego is not None:
        break
    time.sleep(1)

if ego is None:
    raise RuntimeError(
        "Tesla not found. Keep manual_control.py running."
    )

print("Ego:", ego.id, ego.get_location())

walker_bps = list(blueprints.filter("walker.pedestrian.*"))
controller_bp = blueprints.find("controller.ai.walker")

# Sample navigation points across the map.
print("Sampling pedestrian navigation points...")
points = []

for _ in range(8000):
    p = world.get_random_location_from_navigation()
    if p is not None:
        points.append(p)

if not points:
    raise RuntimeError("No pedestrian navigation points found.")

ego_transform = ego.get_transform()
ego_loc = ego_transform.location
forward = ego_transform.get_forward_vector()
right = ego_transform.get_right_vector()

def dot(delta, direction):
    return (
        delta.x * direction.x
        + delta.y * direction.y
        + delta.z * direction.z
    )

# Find points reasonably close and in front of the Tesla.
nearby = []

for p in points:
    delta = p - ego_loc
    distance = p.distance(ego_loc)
    ahead = dot(delta, forward)
    side = dot(delta, right)

    if 4 <= distance <= 45 and ahead > 0:
        nearby.append((p, distance, ahead, side))

if not nearby:
    points.sort(key=lambda p: p.distance(ego_loc))
    print("No nearby point ahead. Closest sampled points:")
    for p in points[:10]:
        print(round(p.distance(ego_loc), 2), p)

    raise RuntimeError(
        "No pedestrian navigation point found in front of the Tesla. "
        "Move the Tesla to another road section and retry."
    )

# Choose a starting point near the road ahead.
nearby.sort(key=lambda item: item[1])
start, start_distance, start_ahead, start_side = nearby[0]

# Prefer a destination on the opposite side of the road.
opposite = []

for p, distance, ahead, side in nearby:
    separation = p.distance(start)

    if separation < 5 or separation > 25:
        continue

    if start_side * side < 0 and abs(ahead - start_ahead) < 10:
        opposite.append((abs(separation - 10), p))

if opposite:
    opposite.sort(key=lambda item: item[0])
    destination = opposite[0][1]
else:
    # Fallback: any distinct nearby navigation point.
    alternatives = [
        p for p, _, _, _ in nearby
        if 5 <= p.distance(start) <= 20
    ]

    if not alternatives:
        raise RuntimeError(
            "Found a start point but no suitable destination. "
            "Try another road section."
        )

    destination = min(
        alternatives,
        key=lambda p: abs(p.distance(start) - 10)
    )

print("Start:", start)
print("Destination:", destination)
print("Start distance from Tesla:", round(start.distance(ego_loc), 2), "m")

walker_bp = random.choice(walker_bps)
walker_bp.set_attribute("is_invincible", "true")

walker = world.try_spawn_actor(
    walker_bp,
    carla.Transform(start)
)

if walker is None:
    raise RuntimeError(
        "Spawn failed. Run again to select another navigation point."
    )

controller = None

try:
    controller = world.try_spawn_actor(
        controller_bp,
        carla.Transform(),
        attach_to=walker
    )

    if controller is None:
        raise RuntimeError("Could not spawn walker controller.")

    time.sleep(1)
    controller.start()
    controller.set_max_speed(1.5)
    controller.go_to_location(destination)

    print("Pedestrian movement started.")
    print("This is a person stand-in, not an animal model.")
    print("Press Ctrl+C to stop.")

    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopping scenario...")

finally:
    if controller is not None:
        try:
            controller.stop()
        except RuntimeError:
            pass
        controller.destroy()

    if walker is not None:
        walker.destroy()

    print("Pedestrian removed.")
