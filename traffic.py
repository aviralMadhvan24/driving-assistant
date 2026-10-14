import carla
import random
import time

HOST = "127.0.0.1"
PORT = 2000

client = carla.Client(HOST, PORT)
client.set_timeout(10.0)

world = client.get_world()
traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_global_distance_to_leading_vehicle(5.0)

blueprints = world.get_blueprint_library().filter("vehicle.*")

spawn_points = world.get_map().get_spawn_points()

vehicles = []

# Spawn 20 vehicles
for i in range(min(20, len(spawn_points))):

    bp = random.choice(blueprints)

    # Avoid spawning another Tesla Model 3 as traffic
    if bp.id == "vehicle.tesla.model3":
        continue

    transform = spawn_points[i]

    vehicle = world.try_spawn_actor(bp, transform)

    if vehicle:
        vehicle.set_autopilot(True, traffic_manager.get_port())
        vehicles.append(vehicle)

print(f"Spawned {len(vehicles)} traffic vehicles.")

try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping traffic...")

finally:
    for vehicle in vehicles:
        vehicle.destroy()

    print("Traffic removed.")