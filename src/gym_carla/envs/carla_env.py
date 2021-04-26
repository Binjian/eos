#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import argparse
import collections
import pygame
import sys
import random
import time
import pyglet
import pandas as pd
from skimage.transform import resize

import xmlmaker
import gym
from gym import spaces
from gym.utils import seeding
import carla
import re
import xml.etree.ElementTree as ET

from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser


try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

try:
    import numpy as np
except ImportError:
    raise RuntimeError("cannot import numpy, make sure numpy package is installed")


def find_weather_presets():
    rgx = re.compile(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    name = lambda x: " ".join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, params):
        # parameters
        self.display_size = params["display_size"]  # rendering screen size
        self.max_past_step = params["max_past_step"]
        self.number_of_vehicles = params["number_of_vehicles"]
        self.number_of_walkers = params["number_of_walkers"]
        self.dt = params["dt"]
        self.task_mode = params["task_mode"]
        self.max_time_episode = params["max_time_episode"]
        self.max_waypt = params["max_waypt"]
        self.obs_range = params["obs_range"]
        self.lidar_bin = params["lidar_bin"]
        self.d_behind = params["d_behind"]
        self.obs_size = int(self.obs_range / self.lidar_bin)
        self.out_lane_thres = params["out_lane_thres"]
        self.desired_speed = params["desired_speed"]
        self.max_ego_spawn_times = params["max_ego_spawn_times"]
        self.display_route = params["display_route"]
        self.ai_mode = params["AI_mode"]
        self.file_path = params["file_path"]

        # driving mode
        self.autoflag = False
        # keyboard manual control settings
        self.clock = pygame.time.Clock()
        self._cache = 0
        self.pedal = 0
        self.steer_increment = 0.5
        self.brake = 0
        self.reverse = False

        # start and end point for race track

        self.start_point = [-40, 34, 3, 0, 0]

        if self.file_path == "../data/highring.xodr":
            self.end_point = [-52, 34, 3, 0, 0]
        elif self.file_path == "../data/straight.xodr":
            self.end_point = [260, 34, 3, 0, 0]

        # target speed
        self.df = pd.read_excel("../data/waypoint_highring.xls")
        self.counter = 0

        # G29 settings
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read("wheel_config.ini")
        self._steer_idx = int(self._parser.get("G29 Racing Wheel", "steering_wheel"))
        self._throttle_idx = int(self._parser.get("G29 Racing Wheel", "throttle"))
        self._brake_idx = int(self._parser.get("G29 Racing Wheel", "brake"))
        self._reverse_idx = int(self._parser.get("G29 Racing Wheel", "reverse"))
        self._handbrake_idx = int(self._parser.get("G29 Racing Wheel", "handbrake"))

        # 0 is autopilot, 1 is reinforcement learning, 2 is manual control
        self.modename = ["AutoPilot", "Reinforcement Learning", "Manual Control"]
        if self.ai_mode:
            self.mode = 1
        else:
            self.mode = 2
        self._mode_transforms = 3

        # create map and define start point of agent
        self.map_road_number = params["map_road_number"]
        self.start = xmlmaker.createmap(self.map_road_number)
        self.circle_num = 1
        self.circle_thre = 3
        self.diff = []

        # info

        if "pixor" in params.keys():
            self.pixor = params["pixor"]
            self.pixor_size = params["pixor_size"]
        else:
            self.pixor = False

        # Destination
        if params["task_mode"] == "roundabout":
            self.dests = [
                [4.46, -61.46, 0],
                [-49.53, -2.89, 0],
                [-6.48, 55.47, 0],
                [35.96, 3.33, 0],
            ]
        else:
            self.dests = None

        # action and observation spaces
        self.discrete = params["discrete"]
        self.discrete_act = [
            params["discrete_acc"],
            params["discrete_steer"],
        ]  # acc, steer
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
        else:
            self.action_space = spaces.Box(
                np.array(
                    [
                        params["continuous_accel_range"][0],
                        params["continuous_steer_range"][0],
                    ]
                ),
                np.array(
                    [
                        params["continuous_accel_range"][1],
                        params["continuous_steer_range"][1],
                    ]
                ),
                dtype=np.float32,
            )  # acc, steer
        observation_space_dict = {
            "camera": spaces.Box(
                low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8
            ),
            "lidar": spaces.Box(
                low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8
            ),
            "birdeye": spaces.Box(
                low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8
            ),
            "newcam": spaces.Box(
                low=0,
                high=511,
                shape=(2 * self.obs_size, 2 * self.obs_size, 3),
                dtype=np.uint8,
            ),
            "newcam2": spaces.Box(
                low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8
            ),
            "state": spaces.Box(
                np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32
            ),
        }
        if self.pixor:
            observation_space_dict.update(
                {
                    "roadmap": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.obs_size, self.obs_size, 3),
                        dtype=np.uint8,
                    ),
                    "vh_clas": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.pixor_size, self.pixor_size, 1),
                        dtype=np.float32,
                    ),
                    "vh_regr": spaces.Box(
                        low=-5,
                        high=5,
                        shape=(self.pixor_size, self.pixor_size, 6),
                        dtype=np.float32,
                    ),
                    "newmap": spaces.Box(
                        low=0,
                        high=511,
                        shape=(2 * self.obs_size, 2 * self.obs_size, 3),
                        dtype=np.uint8,
                    ),
                    "newmap2": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.obs_size, self.obs_size, 3),
                        dtype=np.uint8,
                    ),
                    "pixor_state": spaces.Box(
                        np.array([-1000, -1000, -1, -1, -5]),
                        np.array([1000, 1000, 1, 1, 20]),
                        dtype=np.float32,
                    ),
                }
            )
        self.observation_space = spaces.Dict(observation_space_dict)

        # Connect to carla server and get world object
        print("connecting to Carla server...")
        client = carla.Client(params["carlaserver"], params["port"])
        client.set_timeout(10.0)
        myFile = ET.parse(self.file_path)
        root = myFile.getroot()
        xodrStr = ET.tostring(root, encoding="utf8", method="xml")
        self.world = client.generate_opendrive_world(opendrive=xodrStr)
        # self.world = client.load_world(params['town'])
        print("Carla server connected!")

        # Set weather
        self.world._weather_presets = find_weather_presets()
        self.world._weather_index = 0

        # Get spawn points
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        self.walker_spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc != None:
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(
            params["ego_vehicle_filter"], color="0,0,0"
        )

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )

        # Lidar sensor
        self.lidar_data = None
        self.lidar_height = 2.1
        self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
        self.lidar_bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        self.lidar_bp.set_attribute("channels", "32")
        self.lidar_bp.set_attribute("range", "5000")

        # Camera sensor
        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=1.9, y=-0.3, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")

        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute("image_size_x", str(self.obs_size))
        self.camera_bp.set_attribute("image_size_y", str(self.obs_size))
        self.camera_bp.set_attribute("fov", "110")

        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute("sensor_tick", "0.02")

        # newcam sensor
        self.newcam_img = np.zeros(
            (2 * self.obs_size, 2 * self.obs_size, 3), dtype=np.uint8
        )
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-10.0, z=6.0)),
            carla.Transform(carla.Location(x=1.9, y=-0.3, z=1.7)),
            carla.Transform(carla.Location(x=4, z=1.5)),
            carla.Transform(carla.Location(x=-5.5, z=2.5)),
            carla.Transform(carla.Location(x=5.5, z=80), carla.Rotation(pitch=-90.0)),
        ]
        self.newcam_index = 0
        self.newcam_trans = self._camera_transforms[self.newcam_index]
        self.newcam_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")

        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.newcam_bp.set_attribute("image_size_x", str(2 * self.obs_size))
        self.newcam_bp.set_attribute("image_size_y", str(2 * self.obs_size))
        self.newcam_bp.set_attribute("fov", "90")

        # Set the time in seconds between sensor captures
        self.newcam_bp.set_attribute("sensor_tick", "0.02")

        # newcam2 sensor
        self.newcam2_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.newcam2_trans = carla.Transform(
            carla.Location(x=5.5, z=80), carla.Rotation(pitch=-90.0)
        )
        self.newcam2_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")

        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.newcam2_bp.set_attribute("image_size_x", str(self.obs_size))
        self.newcam2_bp.set_attribute("image_size_y", str(self.obs_size))
        self.newcam2_bp.set_attribute("fov", "110")

        # Set the time in seconds between sensor captures
        self.newcam2_bp.set_attribute("sensor_tick", "0.02")

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # Initialize the renderer
        self._init_renderer()
        self._parse_events()
        print("----------------------------")
        print(self.modename[self.mode])

        # Get pixel grid points
        if self.pixor:
            x, y = np.meshgrid(
                np.arange(self.pixor_size), np.arange(self.pixor_size)
            )  # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            self.pixel_grid = np.vstack((x, y))

    def get_simulation_v(self):
        time_length = len(self.df["v"])
        time_step = range(time_length)
        sim_step = np.arange(0, time_length, 0.1)
        self.v_sim = np.interp(sim_step, time_step, self.df["v"])

        return self.v_sim

    def toggle_mode(self):
        self.mode = (self.mode + 1) % (self._mode_transforms)

    def toggle_camera(self):
        self.newcam_index = (self.newcam_index + 1) % len(self._camera_transforms)
        self.newcam_trans = self._camera_transforms[self.newcam_index]
        self.newcam_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")

        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.newcam_bp.set_attribute("image_size_x", str(2 * self.obs_size))
        self.newcam_bp.set_attribute("image_size_y", str(2 * self.obs_size))
        self.newcam_bp.set_attribute("fov", "110")
        self.newcam_sensor = self.world.spawn_actor(
            self.newcam_bp, self.newcam_trans, attach_to=self.ego
        )
        self.newcam_sensor.listen(lambda data: get_newcam_img(data))

        def get_newcam_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.newcam_img = array

        newcam = resize(self.newcam_img, (2 * self.obs_size, 2 * self.obs_size)) * 255
        newcam_surface = rgb_to_display_surface(newcam, 2 * self.display_size)
        self.display.blit(newcam_surface, (0, 0))

    def next_weather(self, reverse=False):
        self.world._weather_index += -1 if reverse else 1
        self.world._weather_index %= len(self.world._weather_presets)
        preset = self.world._weather_presets[self.world._weather_index]
        self.world.set_weather(preset[0])

    def _parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    self.deleteElement()
                    pygame.quit()
                    sys.exit
                if event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    self.next_weather(reverse=True)
                elif event.key == K_c:
                    self.next_weather()
                elif event.key == K_TAB:
                    self.toggle_camera()
                elif event.key == K_p:
                    self.toggle_mode()
                    print("----------------------------")
                    print(self.modename[self.mode])
                elif event.key == K_EQUALS:
                    self.ai_mode = not self.ai_mode
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    self.reverse = not self.reverse
                elif event.button == 6:
                    self.next_weather()
                elif event.button == 10:
                    self.toggle_camera()
                elif event.button == 7:
                    self.ai_mode = not self.ai_mode

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

    # keyboard simulation of pedal, not used if logitech is available
    def _parse_vehicle_keys(self, keys):

        if keys[K_UP] or keys[K_w]:
            self.pedal = 1.0
        else:
            self.pedal = 0.0

        if keys[K_LEFT] or keys[K_a]:
            self.steer_cache -= self.steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self.steer_cache += self.steer_increment
        else:
            self.steer_cache = 0.0
        self.steer_cache = min(0.7, max(-0.7, self.steer_cache))
        steer = round(self.steer_cache, 1)
        if keys[K_DOWN] or keys[K_s]:
            self.brake = 1.0
        else:
            self.brake = 0.0

        if keys[K_r]:
            self.reverse = not self.reverse

    def _parse_g29_keys(self):

        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [
            float(self._joystick.get_button(i))
            for i in range(self._joystick.get_numbuttons())
        ]
        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is

        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        # throttleCmd = K2 + (2.05 * math.log10(
        #     -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92

        throttleCmd = -(jsInputs[self._throttle_idx] - 1) / 2

        if throttleCmd <= 0.01:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        # brakeCmd = 1.6 + (2.05 * math.log10(
        #     -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        brakeCmd = -(jsInputs[self._brake_idx] - 1) / 2

        if brakeCmd <= 0.01:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        # if jsButtons[self._reverse_idx]:
        #     self.reverse = not self.reverse

        self.steer = steerCmd
        self.brake = brakeCmd
        self.pedal = throttleCmd

        # print(jsButtons)
    def get_init_state(self):
        # State observation
        self._parse_g29_keys()

        v = self.ego.get_velocity()
        a = self.ego.get_acceleration()
        speed = 3.6 * np.sqrt(v.x ** 2 + v.y ** 2)
        acc = np.sqrt(a.x ** 2 + a.y ** 2)

        # return just speed and acceleration
        observation = [speed, acc, self.pedal]
        return observation


    def reset(self):
        # Clear sensor objects
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None
        self.newcam_sensor = None
        self.newcam2_sensor = None
        self.counter = 0
        self.diff = []

        # Delete sensors, vehicles and walkers
        self._clear_all_actors(
            [
                "sensor.other.collision",
                "sensor.lidar.ray_cast",
                "sensor.camera.rgb",
                "vehicle.*",
                "controller.ai.walker",
                "walker.*",
            ]
        )

        # Disable sync mode
        self._set_synchronous_mode(False)

        # Spawn surrounding vehicles
        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_vehicle_at(
                random.choice(self.vehicle_spawn_points), number_of_wheels=[4]
            ):
                count -= 1

        # Spawn pedestrians
        random.shuffle(self.walker_spawn_points)
        count = self.number_of_walkers
        if count > 0:
            for spawn_point in self.walker_spawn_points:
                if self._try_spawn_random_walker_at(spawn_point):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_walker_at(
                random.choice(self.walker_spawn_points)
            ):
                count -= 1

        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons("vehicle.*")
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons("walker.*")
        self.walker_polygons.append(walker_poly_dict)

        # Spawn the ego vehicle
        ego_spawn_times = 0
        while True:
            if ego_spawn_times > self.max_ego_spawn_times:
                self.reset()

            if self.task_mode == "random":
                # transform = random.choice(self.vehicle_spawn_points)
                # transform = self.vehicle_spawn_points[1]
                # # start_point = [
                # #     self.start[0] + 6,
                # #     -self.start[1],
                # #     self.start[2],
                # #     -self.start[3] * 180 / math.pi,
                # #     self.start[4],
                # # ]
                # self.start_point = [-56,34,3,0,0]
                # print(self.start_point)
                transform = set_carla_transform(self.start_point)
            if self.task_mode == "roundabout":
                self.start = [52.1 + np.random.uniform(-5, 5), -4.2, 178.66]  # random
                # self.start= [0,8.823615,1,175.5]
                transform = set_carla_transform(self.start)
                print("spawned")
            if self._try_spawn_ego_vehicle_at(transform):
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(
            self.collision_bp, carla.Transform(), attach_to=self.ego
        )
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_hist = []

        # Add lidar sensor
        self.lidar_sensor = self.world.spawn_actor(
            self.lidar_bp, self.lidar_trans, attach_to=self.ego
        )
        self.lidar_sensor.listen(lambda data: get_lidar_data(data))

        def get_lidar_data(data):
            self.lidar_data = data

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(
            self.camera_bp, self.camera_trans, attach_to=self.ego
        )
        self.camera_sensor.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

        # Add newcam sensor
        self.newcam_sensor = self.world.spawn_actor(
            self.newcam_bp, self.newcam_trans, attach_to=self.ego
        )
        self.newcam_sensor.listen(lambda data: get_newcam_img(data))

        def get_newcam_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.newcam_img = array

        # Add camera sensor
        self.newcam2_sensor = self.world.spawn_actor(
            self.newcam2_bp, self.newcam2_trans, attach_to=self.ego
        )
        self.newcam2_sensor.listen(lambda data: get_newcam2_img(data))

        def get_newcam2_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.newcam2_img = array

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # Set ego information for render
        self.birdeye_render.set_hero(self.ego, self.ego.id)

        return self._get_obs()

    def deleteElement(self):
        # Clear sensor objects
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None
        self.newcam_sensor = None
        self.newcam2_sensor = None

        # Delete sensors, vehicles and walkers
        self._clear_all_actors(
            [
                "sensor.other.collision",
                "sensor.lidar.ray_cast",
                "sensor.camera.rgb",
                "vehicle.*",
                "controller.ai.walker",
                "walker.*",
            ]
        )

    def step(self, action):
        if self.mode == 1:
            self.autoflag = False
            self.ego.set_autopilot(self.autoflag)
            # Calculate acceleration and steering
            if self.discrete:
                acc = self.discrete_act[0][action // self.n_steer]
                steer = self.discrete_act[1][action % self.n_steer]
            else:
                acc = action[0]
                steer = self.steer

            # Convert acceleration to throttle and brake
            if acc > 0:
                throttle = np.clip(acc, 0, 1)
                brake = 0
            else:
                throttle = 0
                brake = np.clip(-acc, 0, 1)
            # Apply control
            act = carla.VehicleControl(
                throttle=float(throttle), steer=float(-steer), brake=float(brake)
            )
            self.ego.apply_control(act)
            self.world.tick()
            self.counter += 1

        elif self.mode == 0:
            self.autoflag = True
            self.ego.set_autopilot(self.autoflag)
            self.world.tick()

        elif self.mode == 2:
            self.autoflag = False
            self.ego.set_autopilot(self.autoflag)
            act = carla.VehicleControl(
                throttle=self.pedal,
                steer=self.steer,
                brake=self.brake,
                reverse=self.reverse,
            )
            self.ego.apply_control(act)
            # self._parse_vehicle_keys(pygame.key.get_pressed())
            self.world.tick()

        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons("vehicle.*")
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)
        walker_poly_dict = self._get_actor_polygons("walker.*")
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
            self.walker_polygons.pop(0)

        # route planner
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # state information
        info = {"waypoints": self.waypoints, "vehicle_front": self.vehicle_front}

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        return (
            self._get_obs(),
            self._get_reward(),
            self._terminal(),
            copy.deepcopy(info),
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode):
        pass

    def _create_vehicle_bluepprint(self, actor_filter, color, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.

        Args:
          actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
          bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [
                x for x in blueprints if int(x.get_attribute("number_of_wheels")) == nw
            ]
        bp = random.choice(blueprint_library)
        # bp.set_attribute("color",color)

        return bp

    def _init_renderer(self):
        """Initialize the birdeye view renderer."""
        # full screen
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (self.display_size * 4, self.display_size * 2), pygame.FULLSCREEN, 32
        )
        # for small screen uncomment below
        self.display = pygame.display.set_mode(
            (self.display_size * 4, self.display_size * 2),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        pixels_per_meter = self.display_size / self.obs_range
        pixels_ahead_vehicle = (self.obs_range / 2 - self.d_behind) * pixels_per_meter
        birdeye_params = {
            "screen_size": [2 * self.display_size, 2 * self.display_size],
            "pixels_per_meter": pixels_per_meter,
            "pixels_ahead_vehicle": pixels_ahead_vehicle,
        }
        self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode."""
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.

        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint(
            "vehicle.*", number_of_wheels=number_of_wheels
        )
        blueprint.set_attribute("role_name", "autopilot")
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot()
            return True
        return False

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.

        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(self.world.get_blueprint_library().filter("walker.*"))
        # set as not invencible
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "False")
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find(
                "controller.ai.walker"
            )
            walker_controller_actor = self.world.spawn_actor(
                walker_controller_bp, carla.Transform(), walker_actor
            )
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(
                self.world.get_random_location_from_navigation()
            )
            # random max speed
            walker_controller_actor.set_max_speed(
                1 + random.random()
            )  # max speed between 1 and 2 (default is 1.4 m/s)
            return True
        return False

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego = vehicle
            return True

        return False

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
          filt: the filter indicating what type of actors we'll look at.

        Returns:
          actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _get_obs(self):
        """Get the observations."""
        ## Birdeye rendering
        self.birdeye_render.vehicle_polygons = self.vehicle_polygons
        self.birdeye_render.walker_polygons = self.walker_polygons
        self.birdeye_render.waypoints = self.waypoints

        # birdeye view with roadmap and actors
        birdeye_render_types = ["roadmap", "actors"]
        if self.display_route:
            birdeye_render_types.append("waypoints")
        self.birdeye_render.render(self.display, birdeye_render_types)
        birdeye = pygame.surfarray.array3d(self.display)
        birdeye = birdeye[0 : 2 * self.display_size, :, :]
        birdeye = display_to_rgb(birdeye, self.obs_size)

        # Roadmap
        if self.pixor:
            roadmap_render_types = ["roadmap"]
            if self.display_route:
                roadmap_render_types.append("waypoints")
            self.birdeye_render.render(self.display, roadmap_render_types)
            roadmap = pygame.surfarray.array3d(self.display)
            roadmap = roadmap[0 : self.display_size, :, :]
            roadmap = display_to_rgb(roadmap, self.obs_size)
            # Add ego vehicle
            for i in range(self.obs_size):
                for j in range(self.obs_size):
                    if (
                        abs(birdeye[i, j, 0] - 255) < 20
                        and abs(birdeye[i, j, 1] - 0) < 20
                        and abs(birdeye[i, j, 0] - 255) < 20
                    ):
                        roadmap[i, j, :] = birdeye[i, j, :]

            # newmap
            if self.pixor:
                newmap_render_types = ["roadmap"]
                if self.display_route:
                    newmap_render_types.append("waypoints")
                self.birdeye_render.render(self.display, newmap_render_types)
                newmap = pygame.surfarray.array3d(self.display)
                newmap = newmap[0 : 2 * self.display_size, :, :]
                newmap = display_to_rgb(2 * newmap, self.obs_size)
                # Add ego vehicle
                for i in range(self.obs_size):
                    for j in range(self.obs_size):
                        if (
                            abs(birdeye[i, j, 0] - 255) < 20
                            and abs(birdeye[i, j, 1] - 0) < 20
                            and abs(birdeye[i, j, 0] - 255) < 20
                        ):
                            newmap[i, j, :] = birdeye[i, j, :]

            # newmap2
            if self.pixor:
                newmap2_render_types = ["roadmap"]
                if self.display_route:
                    newmap2_render_types.append("waypoints")
                self.birdeye_render.render(self.display, newmap2_render_types)
                newmap2 = pygame.surfarray.array3d(self.display)
                newmap2 = newmap2[0 : self.display_size, :, :]
                newmap2 = display_to_rgb(newmap2, self.obs_size)
                # Add ego vehicle
                for i in range(self.obs_size):
                    for j in range(self.obs_size):
                        if (
                            abs(birdeye[i, j, 0] - 255) < 20
                            and abs(birdeye[i, j, 1] - 0) < 20
                            and abs(birdeye[i, j, 0] - 255) < 20
                        ):
                            newmap[i, j, :] = birdeye[i, j, :]

        # Lidar image generation
        location = np.frombuffer(self.lidar_data.raw_data, dtype=np.dtype("f4"))
        points = np.reshape(location, (-1, 4))
        points = points[:, :-1]
        point_cloud = points.copy()
        point_cloud[:, 0] = points[:, 1]
        point_cloud[:, 1] = -points[:, 0]

        # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin,
        # and z is set to be two bins.
        y_bins = np.arange(
            -(self.obs_range - self.d_behind),
            self.d_behind + self.lidar_bin,
            self.lidar_bin,
        )
        x_bins = np.arange(
            -self.obs_range / 2, self.obs_range / 2 + self.lidar_bin, self.lidar_bin
        )
        z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.25, 1]
        # Get lidar image according to the bins
        lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
        lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
        lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
        # Add the waypoints to lidar image
        if self.display_route:
            wayptimg = (
                (birdeye[:, :, 0] <= 10)
                * (birdeye[:, :, 1] <= 10)
                * (birdeye[:, :, 2] >= 240)
            )
        else:
            wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
        wayptimg = np.expand_dims(wayptimg, axis=2)
        wayptimg = np.fliplr(np.rot90(wayptimg, 3))

        # Get the final lidar image
        lidar = np.concatenate((lidar, wayptimg), axis=2)
        lidar = np.flip(lidar, axis=1)
        lidar = np.rot90(lidar, 1)
        lidar = lidar * 255

        # Display birdeye image
        birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
        # self.display.blit(birdeye_surface, (self.display_size * 2, self.display_size))
        text_color = (255, 255, 255)
        v_color = (0, 255, 0)
        v_target_color = (255, 20, 147)
        c_warning = (255, 0, 0)
        background = (0, 0, 0)
        font = pygame.font.Font("freesansbold.ttf", 16)
        fontAI = pygame.font.Font("freesansbold.ttf", 20)
        fontSpeed = pygame.font.Font("freesansbold.ttf", 20)
        v = self.ego.get_velocity()
        a = self.ego.get_acceleration()
        speed = 3.6 * np.sqrt(v.x ** 2 + v.y ** 2)
        acc = np.sqrt(a.x ** 2 + a.y ** 2)
        energy = acc ** 2

        v_target_list = self.get_simulation_v()
        v_target = v_target_list[self.counter]
        self.diff.append(abs(v_target - speed))
        avg_diff = sum(self.diff) / len(self.diff)

        self._title_text = "Information board"
        self._circle_rem = "Circle number: " + str(self.circle_num)
        self._warn_text = "PLease press L2 for AI mode"
        self._v_text = "Speed:   % .3g km/h" % (int(speed))
        self._v_target_text = "Target speed:   % .3g km/h" % (int(v_target))

        if self.ai_mode:
            self._ai_text = "AI mode: On"
        else:
            self._ai_text = "Manual mode: On"

        self._ai_text = fontAI.render(str(self._ai_text), True, v_color, background)
        self._circle_rem = font.render(
            str(self._circle_rem), True, text_color, background
        )
        self._warn_text = font.render(str(self._warn_text), True, c_warning, background)
        self._v_text = fontSpeed.render(str(self._v_text), True, v_color, background)
        self._v_target_text = fontSpeed.render(
            str(self._v_target_text), True, v_target_color, background
        )
        self._title_text = font.render(
            str(self._title_text), True, text_color, background
        )
        self.display.fill(background)
        self.display.blit(
            self._title_text, (self.display_size * 2, self.display_size + 16)
        )
        self.display.blit(
            self._circle_rem, (self.display_size * 2, self.display_size + 64)
        )
        self.display.blit(
            self._ai_text, (self.display_size * 2, self.display_size + 96)
        )
        self.display.blit(
            self._v_text, (self.display_size * 2, self.display_size + 128)
        )
        self.display.blit(
            self._v_target_text, (self.display_size * 2, self.display_size + 160)
        )

        if self.circle_num == 2 and self.ai_mode == False:
            self.display.blit(
                self._warn_text, (self.display_size * 2, self.display_size + 224)
            )

        # Display lidar image
        lidar_surface = rgb_to_display_surface(lidar, self.display_size)
        self.display.blit(lidar_surface, (self.display_size * 2, 0))

        ## Display camera image
        camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
        camera_surface = rgb_to_display_surface(camera, self.display_size)
        self.display.blit(camera_surface, (self.display_size * 3, 0))

        ## Display newcam image
        newcam = resize(self.newcam_img, (2 * self.obs_size, 2 * self.obs_size)) * 255
        newcam_surface = rgb_to_display_surface(newcam, 2 * self.display_size)
        self.display.blit(newcam_surface, (0, 0))

        ## Display newcam2 image
        newcam2 = resize(self.newcam2_img, (self.obs_size, self.obs_size)) * 255
        newcam2_surface = rgb_to_display_surface(newcam2, self.display_size)
        self.display.blit(newcam2_surface, (self.display_size * 3, self.display_size))

        if self._parse_events():
            return
        # Display on pygame
        pygame.display.update()
        pygame.display.flip()

        # State observation
        self._parse_g29_keys()
        # TODO offset between V_real and V_WLTC
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        delta_yaw = np.arcsin(
            np.cross(w, np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)])))
        )
        state = np.array(
            [
                lateral_dis,
                -delta_yaw,
                speed,
                self.vehicle_front,
                acc,
                avg_diff,
                self.pedal,
            ]
        )

        # return just speed and acceleration
        observation = [speed, acc, self.pedal]
        return observation

        # info display

        # if self.pixor:
        #     ## Vehicle classification and regression maps (requires further normalization)
        #     vh_clas = np.zeros((self.pixor_size, self.pixor_size))
        #     vh_regr = np.zeros((self.pixor_size, self.pixor_size, 6))
        #
        #     # Generate the PIXOR image. Note in CARLA it is using left-hand coordinate
        #     # Get the 6-dim geom parametrization in PIXOR, here we use pixel coordinate
        #     for actor in self.world.get_actors().filter("vehicle.*"):
        #         x, y, yaw, l, w = get_info(actor)
        #         x_local, y_local, yaw_local = get_local_pose(
        #             (x, y, yaw), (ego_x, ego_y, ego_yaw)
        #         )
        #         if actor.id != self.ego.id:
        #             if (
        #                 abs(y_local) < self.obs_range / 2 + 1
        #                 and x_local < self.obs_range - self.d_behind + 1
        #                 and x_local > -self.d_behind - 1
        #             ):
        #                 x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = get_pixel_info(
        #                     local_info=(x_local, y_local, yaw_local, l, w),
        #                     d_behind=self.d_behind,
        #                     obs_range=self.obs_range,
        #                     image_size=self.pixor_size,
        #                 )
        #                 cos_t = np.cos(yaw_pixel)
        #                 sin_t = np.sin(yaw_pixel)
        #                 logw = np.log(w_pixel)
        #                 logl = np.log(l_pixel)
        #                 pixels = get_pixels_inside_vehicle(
        #                     pixel_info=(x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel),
        #                     pixel_grid=self.pixel_grid,
        #                 )
        #                 for pixel in pixels:
        #                     vh_clas[pixel[0], pixel[1]] = 1
        #                     dx = x_pixel - pixel[0]
        #                     dy = y_pixel - pixel[1]
        #                     vh_regr[pixel[0], pixel[1], :] = np.array(
        #                         [cos_t, sin_t, dx, dy, logw, logl]
        #                     )
        #
        #     # Flip the image matrix so that the origin is at the left-bottom
        #     vh_clas = np.flip(vh_clas, axis=0)
        #     vh_regr = np.flip(vh_regr, axis=0)
        #
        #     # Pixor statobs[2]e, [x, y, cos(yaw), sin(yaw), speed]
        #     pixor_state = [ego_x, ego_y, np.cos(ego_yaw), np.sin(ego_yaw), speed]
        #
        # obs = {
        #     "camera": camera.astype(np.uint8),
        #     "lidar": lidar.astype(np.uint8),
        #     "birdeye": birdeye.astype(np.uint8),
        #     "newcam": newcam.astype(np.uint8),
        #     "newcam2": newcam2.astype(np.uint8),
        #     "state": state,
        # }
        #
        # if self.pixor:
        #     obs.update(
        #         {
        #             "roadmap": roadmap.astype(np.uint8),
        #             "vh_clas": np.expand_dims(vh_clas, -1).astype(np.float32),
        #             "vh_regr": vh_regr.astype(np.float32),
        #             "newmap": newmap.astype(np.uint8),
        #             "newmap2": newmap2.astype(np.uint8),
        #             "pixor_state": pixor_state,
        #         }
        #     )
        #
        # return obs

    def _get_reward(self):
        """Calculate the step reward."""
        a = self.ego.get_acceleration()
        acc = np.sqrt(a.x ** 2 + a.y ** 2)
        r_engy_consump = (acc ** 2)
        # r_time_lapse = -1  # for fixed trip length consider + r_time_lapse
        # r_trip_length = wp_distance[frame]  # for fixed time range consider + r_trip_length
        reward = -r_engy_consump  # + r_time_lapse
        # TODO add speed as reward (being fast should be  rewarded)

        return reward

        # """Calculate the step reward."""
        # # reward for speed tracking
        v = self.ego.get_velocity()
        # speed = np.sqrt(v.x ** 2 + v.y ** 2)
        # r_speed = -abs(speed - self.desired_speed)
        #
        # # reward for collision
        # r_collision = 0
        # if len(self.collision_hist) > 0:
        #     r_collision = -1
        #
        # # reward for steering:
        # r_steer = -self.ego.get_control().steer ** 2
        #
        # # reward for out of lane
        # ego_x, ego_y = get_pos(self.ego)
        # dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        # r_out = 0
        # if abs(dis) > self.out_lane_thres:
        #     r_out = -1
        #
        # # longitudinal speed
        # lspeed = np.array([v.x, v.y])
        # lspeed_lon = np.dot(lspeed, w)
        #
        # # cost for too fast
        # r_fast = 0
        # if lspeed_lon > self.desired_speed:
        #     r_fast = -1
        #
        # # cost for lateral acceleration
        # r_lat = -abs(self.ego.get_control().steer) * lspeed_lon ** 2
        #
        # r = (
        #     200 * r_collision
        #     + 1 * lspeed_lon
        #     + 10 * r_fast
        #     + 1 * r_out
        #     + r_steer * 5
        #     + 0.2 * r_lat
        #     - 0.1
        # )
        #
        # return r

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = get_pos(self.ego)

        finish_distance = np.sqrt(
            (ego_x - self.end_point[0]) ** 2 + (ego_y - self.end_point[1]) ** 2
        )

        # only terminal condition is to reach finishing line within 10 m
        if finish_distance < 10:
            return True

        # # If collides
        # if len(self.collision_hist) > 0:
        #     return True
        #
        # # If reach maximum timestep
        # if self.time_step > self.max_time_episode:
        #     return True
        #
        # # If at destination
        # if self.dests is not None:  # If at destination
        #     for dest in self.dests:
        #         if np.sqrt((ego_x - dest[0]) ** 2 + (ego_y - dest[1]) ** 2) < 4:
        #             return True
        #
        # # If out of lane
        # dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        # if abs(dis) > self.out_lane_thres:
        #     return True

        return False

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == "controller.ai.walker":
                        actor.stop()
                    actor.destroy()
