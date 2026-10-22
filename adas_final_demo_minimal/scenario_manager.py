
import carla
import random
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

HOST = "127.0.0.1"
PORT = 2000
CONTROL_PORT = 8010
TM_PORT = 8000

managed_actors = []
scenario_name = "Night Traffic"
scenario_lock = threading.Lock()


def connect():
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    return client, client.get_world()


client, world = connect()
traffic_manager = client.get_trafficmanager(TM_PORT)
blueprints = world.get_blueprint_library()


def find_ego():
    for vehicle in world.get_actors().filter("vehicle.*"):
        if vehicle.attributes.get("role_name") == "hero":
            return vehicle
    return None


def set_night():
    weather = world.get_weather()
    weather.sun_altitude_angle = -15.0
    weather.cloudiness = 15.0
    weather.precipitation = 0.0
    weather.fog_density = 5.0
    world.set_weather(weather)


def clear_scenario():
    global managed_actors

    for actor in managed_actors:
        try:
            if actor.is_alive:
                if actor.type_id.startswith("vehicle."):
                    actor.set_autopilot(False, TM_PORT)
                actor.destroy()
        except Exception:
            pass

    managed_actors = []


def spawn_vehicle(location, yaw, autopilot=True):
    choices = [
        bp for bp in blueprints.filter("vehicle.*")
        if bp.id != "vehicle.tesla.model3"
    ]
    if not choices:
        print("No vehicle blueprints found.")
        return None

    bp = random.choice(choices)
    transform = carla.Transform(
        location + carla.Location(z=0.5),
        carla.Rotation(yaw=yaw)
    )

    actor = world.try_spawn_actor(bp, transform)
    if actor:
        managed_actors.append(actor)
        if autopilot:
            actor.set_autopilot(True, TM_PORT)
            try:
                traffic_manager.auto_lane_change(actor, True)
                traffic_manager.vehicle_percentage_speed_difference(
                    actor, random.uniform(0, 20)
                )
            except Exception:
                pass
        print(
            f"Spawned {'moving' if autopilot else 'static'} vehicle: "
            f"{actor.type_id}"
        )
    else:
        print("Vehicle spawn failed; position may be occupied.")

    return actor


def spawn_pedestrian(location):
    choices = blueprints.filter("walker.pedestrian.*")
    if not choices:
        print("No pedestrian blueprints found.")
        return None

    bp = random.choice(choices)
    actor = world.try_spawn_actor(
        bp,
        carla.Transform(location + carla.Location(z=0.2))
    )

    if actor:
        managed_actors.append(actor)
        print("Spawned pedestrian obstacle:", actor.type_id)
    else:
        print("Pedestrian spawn failed.")

    return actor


def get_positions():
    ego = find_ego()
    if ego is None:
        raise RuntimeError(
            "Ego vehicle not found. Start manual_control.py first."
        )

    transform = ego.get_transform()
    forward = transform.get_forward_vector()
    right = transform.get_right_vector()
    base = transform.location
    yaw = transform.rotation.yaw

    def ahead(distance, side=0.0):
        return carla.Location(
            x=base.x + forward.x * distance + right.x * side,
            y=base.y + forward.y * distance + right.y * side,
            z=base.z
        )

    return ahead, yaw


def run_scenario(number):
    global scenario_name

    with scenario_lock:
        clear_scenario()
        set_night()

        ahead, yaw = get_positions()

        scenarios = {
            1: "Night Traffic",
            2: "Vehicle Ahead",
            3: "Pedestrian Ahead",
            4: "Close Vehicle Obstacle",
            5: "Night Fog",
            6: "Dense Night Traffic",
        }
        scenario_name = scenarios[number]

        if number == 1:
            spawn_vehicle(ahead(25), yaw)
            spawn_vehicle(ahead(40, 3.0), yaw)

        elif number == 2:
            spawn_vehicle(ahead(22), yaw)

        elif number == 3:
            spawn_vehicle(ahead(28), yaw)
            spawn_pedestrian(ahead(12, -2.0))

        elif number == 4:
            # Close obstacle is deliberately static for this test.
            spawn_vehicle(ahead(10), yaw, autopilot=False)

        elif number == 5:
            weather = world.get_weather()
            weather.sun_altitude_angle = -20.0
            weather.cloudiness = 50.0
            weather.fog_density = 45.0
            weather.precipitation = 0.0
            world.set_weather(weather)
            spawn_vehicle(ahead(22), yaw)

        elif number == 6:
            spawn_vehicle(ahead(18), yaw)
            spawn_vehicle(ahead(30, 3.0), yaw)
            spawn_vehicle(ahead(42, -3.0), yaw)
            spawn_pedestrian(ahead(15, -2.0))

        print(f"\nACTIVE SCENARIO: {scenario_name}\n")


def reset_default():
    run_scenario(1)


class ControlHandler(BaseHTTPRequestHandler):
    def respond(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.respond(200, {"ok": True})

    def do_GET(self):
        self.respond(200, {
            "active_scenario": scenario_name,
            "available_scenarios": [
                "Night Traffic",
                "Vehicle Ahead",
                "Pedestrian Ahead",
                "Close Vehicle Obstacle",
                "Night Fog",
                "Dense Night Traffic",
            ]
        })

    def do_POST(self):
        if self.path not in ("/scenario", "/reset"):
            self.respond(404, {"error": "Unknown endpoint"})
            return

        try:
            if self.path == "/reset":
                reset_default()
                self.respond(200, {
                    "ok": True,
                    "active_scenario": scenario_name
                })
                return

            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            number = int(data.get("scenario", 1))

            if number not in range(1, 7):
                self.respond(400, {"error": "Scenario must be 1-6"})
                return

            run_scenario(number)
            self.respond(200, {
                "ok": True,
                "active_scenario": scenario_name
            })

        except Exception as error:
            self.respond(500, {"error": str(error)})

    def log_message(self, format_string, *args):
        pass


def start_control_server():
    server = HTTPServer(("127.0.0.1", CONTROL_PORT), ControlHandler)
    print(f"Scenario control API: http://127.0.0.1:{CONTROL_PORT}")
    server.serve_forever()


def main():
    set_night()

    print("\n=== CARLA NIGHT ADAS DEMO ===")
    print("1 Night Traffic | 2 Vehicle Ahead")
    print("3 Pedestrian Ahead | 4 Close Obstacle")
    print("5 Night Fog | 6 Dense Night Traffic")
    print("R Reset | Q Quit")
    print("HTTP control is also enabled for dashboard integration.")

    threading.Thread(
        target=start_control_server,
        daemon=True
    ).start()

    reset_default()

    while True:
        try:
            command = input("\nScenario > ").strip().lower()

            if command == "q":
                with scenario_lock:
                    clear_scenario()
                print("Manager closed; CARLA remains running.")
                break
            elif command == "r":
                reset_default()
            elif command in {"1", "2", "3", "4", "5", "6"}:
                run_scenario(int(command))
            else:
                print("Enter 1-6, R, or Q.")

        except KeyboardInterrupt:
            with scenario_lock:
                clear_scenario()
            break
        except Exception as error:
            print("Scenario error:", error)


if __name__ == "__main__":
    main()
