
import carla

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10)

world = client.get_world()

world.set_weather(
    carla.WeatherParameters(
        sun_altitude_angle=-90,
        sun_azimuth_angle=0,
        cloudiness=30,
        precipitation=0,
        fog_density=5,
    )
)

print("Night mode enabled successfully.")
print("Map:", world.get_map().name)
