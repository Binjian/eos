#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla

import argparse
import random
import time

red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)

trail_life_time = 0
waypoint_separation = 6


def _lateral_shift(transform, shift, index):
    transform.rotation.yaw += 90
    if index == 1:
        return transform.location + shift * transform.get_forward_vector()
    if index == 2:
        return (
            transform.location
            + shift * transform.get_forward_vector()
            + carla.Location(x=4)
        )


def draw_transform(debug, trans, col=carla.Color(255, 0, 0), lt=0):
    debug.draw_arrow(
        trans.location,
        trans.location + trans.get_forward_vector(),
        thickness=0.3,
        arrow_size=0.1,
        color=col,
        life_time=lt,
    )


def draw_waypoint_union(debug, w0, w1, color=carla.Color(255, 0, 0), lt=0, index=0):
    if index == 0:
        debug.draw_line(
            # w0.transform.location + carla.Location(y=-1.85, z=0.005),
            # w1.transform.location + carla.Location(y=-1.85, z=0.005),
            _lateral_shift(w0.transform, -w0.lane_width * 0.5, 1),
            _lateral_shift(w1.transform, -w1.lane_width * 0.5, 1),
            thickness=0.3,
            color=color,
            life_time=lt,
            persistent_lines=False,
        )
    if index == 1:
        debug.draw_line(
            _lateral_shift(w0.transform, w0.lane_width * 0.5, 1),
            _lateral_shift(w1.transform, w1.lane_width * 0.5, 1),
            thickness=0.3,
            color=color,
            life_time=lt,
            persistent_lines=False,
        )
    if index == 2:
        debug.draw_line(
            _lateral_shift(w0.transform, -w0.lane_width * 0.5, 1),
            _lateral_shift(w1.transform, -w1.lane_width * 0.5, 1),
            thickness=0.3,
            color=color,
            life_time=lt,
            persistent_lines=False,
        )
        debug.draw_line(
            _lateral_shift(w0.transform, w0.lane_width * 0.5, 1),
            _lateral_shift(w1.transform, w1.lane_width * 0.5, 1),
            thickness=0.3,
            color=color,
            life_time=lt,
            persistent_lines=False,
        )
    # debug.draw_point(w1.transform.location + carla.Location(z=0.005), 0.1, color, lt, False)


def draw_waypoint_info(debug, w, lt=0):
    w_loc = w.transform.location
    debug.draw_string(
        w_loc + carla.Location(z=0.5), "lane: " + str(w.lane_id), False, yellow, lt
    )
    debug.draw_string(
        w_loc + carla.Location(z=1.0), "road: " + str(w.road_id), False, blue, lt
    )
    debug.draw_string(
        w_loc + carla.Location(z=-0.5), str(w.lane_change), False, red, lt
    )


def draw_junction(debug, junction, l_time=0):
    """Draws a junction bounding box and the initial and final waypoint of every lane."""
    # draw bounding box
    box = junction.bounding_box
    point1 = box.location + carla.Location(x=box.extent.x, y=box.extent.y, z=0.005)
    point2 = box.location + carla.Location(x=-box.extent.x, y=box.extent.y, z=0.005)
    point3 = box.location + carla.Location(x=-box.extent.x, y=-box.extent.y, z=0.005)
    point4 = box.location + carla.Location(x=box.extent.x, y=-box.extent.y, z=0.005)
    debug.draw_line(
        point1,
        point2,
        thickness=0.1,
        color=orange,
        life_time=l_time,
        persistent_lines=False,
    )
    debug.draw_line(
        point2,
        point3,
        thickness=0.1,
        color=orange,
        life_time=l_time,
        persistent_lines=False,
    )
    debug.draw_line(
        point3,
        point4,
        thickness=0.1,
        color=orange,
        life_time=l_time,
        persistent_lines=False,
    )
    debug.draw_line(
        point4,
        point1,
        thickness=0.1,
        color=orange,
        life_time=l_time,
        persistent_lines=False,
    )
    # draw junction pairs (begin-end) of every lane
    junction_w = junction.get_waypoints(carla.LaneType.Any)
    for pair_w in junction_w:
        # draw_transform(debug, pair_w[0].transform, orange, l_time)
        # debug.draw_point(
        #     pair_w[0].transform.location + carla.Location(z=0.005), 0.1, orange, l_time, False)
        # draw_transform(debug, pair_w[1].transform, orange, l_time)
        # debug.draw_point(
        #     pair_w[1].transform.location + carla.Location(z=0.005), 0.1, orange, l_time, False)
        debug.draw_line(
            pair_w[0].transform.location + carla.Location(z=0.005),
            pair_w[1].transform.location + carla.Location(z=0.005),
            0.1,
            white,
            l_time,
            False,
        )


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--host",
        metavar="H",
        default="192.168.60.80",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    argparser.add_argument(
        "-i", "--info", action="store_true", help="Show text information"
    )
    argparser.add_argument(
        "-x", default= -60, type=float, help="X start position (default: 0.0)"
    )
    argparser.add_argument(
        "-y", default= 36, type=float, help="Y start position (default: 0.0)"
    )
    argparser.add_argument(
        "-z", default=0, type=float, help="Z start position (default: 0.0)"
    )
    argparser.add_argument(
        "-s",
        "--seed",
        metavar="S",
        default=os.getpid(),
        type=int,
        help="Seed for the random path (default: program pid)",
    )
    argparser.add_argument(
        "-t",
        "--tick-time",
        metavar="T",
        default=0.1,
        type=float,
        help="Tick time between updates (forward velocity) (default: 0.2)",
    )
    args = argparser.parse_args()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        world = client.get_world()
        m = world.get_map()
        debug = world.debug
        print(debug)

        random.seed(args.seed)
        print("Seed: ", args.seed)

        loc = carla.Location(args.x, args.y, args.z)
        print("Initial location: ", loc)
        print(loc.x)
        print(loc.y)
        print(loc.z)

        current_w = m.get_waypoint(loc)
        print(current_w.transform)
        current_w_r = current_w.get_right_lane()
        current_w_l = current_w.get_left_lane()
        i = 1
        # main loop
        while True:

            next_w = current_w.next(waypoint_separation)[-1]
            # print(current_w_r)
            next_w_r = current_w_r.next(waypoint_separation)[-1]
            # print(current_w_l)
            next_w_l = current_w_l.next(waypoint_separation)[-1]

            # potential_w = list(current_w.next(waypoint_separation))
            # current_w_r = current_w.get_right_lane()
            # print(current_w_r)
            # w_r = list(current_w_r.next(waypoint_separation))
            # current_w_l = current_w.get_left_lane()
            # print(current_w_l)
            # w_l = list(current_w_l.next(waypoint_separation))

            # # check for available right driving lanes
            # if current_w.lane_change & carla.LaneChange.Right:
            #     right_w = current_w.get_right_lane()
            #     if right_w and right_w.lane_type == carla.LaneType.Driving:
            #         potential_w += list(right_w.next(waypoint_separation))

            # # check for available left driving lanes
            # if current_w.lane_change & carla.LaneChange.Left:
            #     left_w = current_w.get_left_lane()
            #     if left_w and left_w.lane_type == carla.LaneType.Driving:
            #         potential_w += list(left_w.next(waypoint_separation))

            # choose a random waypoint to be the next
            # next_w = random.choice(potential_w)
            # potential_w.remove(next_w)

            # next_w_r = random.choice(w_r)
            # w_r.remove(next_w_r)

            # next_w_l = random.choice(w_l)
            # w_l.remove(next_w_l)
            # Render some nice information, notice that you can't see the strings if you are using an editor camera
            # if args.info:
            #     draw_waypoint_info(debug, current_w, trail_life_time)
            draw_waypoint_union(
                debug,
                current_w,
                next_w,
                cyan if current_w.is_junction else white,
                trail_life_time,
                2,
            )
            # draw_waypoint_union(debug, current_w_r, next_w_r, cyan if current_w.is_junction else white, trail_life_time, 1)
            # draw_waypoint_union(debug, current_w_l, next_w_l, cyan if current_w.is_junction else white, trail_life_time, 0)
            # draw_transform(debug, current_w.transform, white, trail_life_time)

            # print the remaining waypoints
            # for p in potential_w:
            #     draw_waypoint_union(debug, current_w, p, red, trail_life_time)
            #     draw_transform(debug, p.transform, white, trail_life_time)

            # draw all junction waypoints and bounding box
            if next_w.is_junction:
                junction = next_w.get_junction()
                draw_junction(debug, junction, trail_life_time)

            # update the current waypoint and sleep for some time
            current_w = next_w.next(waypoint_separation)[-1]
            current_w_r = next_w_r
            current_w_l = next_w_l
            # time.sleep(args.tick_time)
            i = i + 1

    finally:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExit by user.")
    finally:
        print("\nExit.")
