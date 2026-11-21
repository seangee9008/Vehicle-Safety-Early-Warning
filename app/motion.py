"""Plain descriptions of a Waymo demo example, from its own paths: "Car turning left · 34 km/h"."""

from __future__ import annotations

import math

TYPE_WORD = {"vehicle": "Car", "pedestrian": "Pedestrian", "cyclist": "Cyclist"}
TYPE_ICON = {"vehicle": ":material/directions_car:", "pedestrian": ":material/directions_walk:",
             "cyclist": ":material/directions_bike:"}
TURN_DEG = 25.0          # final heading of the true future, left (+) or right (-) of straight ahead
STOPPED_KMH = 1.0


def _valid(points) -> list[list[float]]:
    return [p for p in points if p is not None]


def speed_kmh(past_path) -> float:
    """Speed now, from the last two past positions (10 per second)."""
    pts = _valid(past_path)
    if len(pts) < 2:
        return 0.0
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    return math.hypot(x1 - x0, y1 - y0) * 10 * 3.6


def movement(past_path, true_future) -> str:
    future = _valid(true_future)
    now = speed_kmh(past_path)
    if not future:
        return "stopped" if now < STOPPED_KMH else "going straight"
    travelled = math.hypot(*future[-1])
    if now < STOPPED_KMH:
        return "stopped" if travelled < 1.0 else "pulling away"
    # heading over the last second of what really happened
    tail = future[-10:] if len(future) >= 10 else future
    (ax, ay), (bx, by) = tail[0], tail[-1]
    end_speed = math.hypot(bx - ax, by - ay) / max(len(tail) - 1, 1) * 10 * 3.6
    angle = math.degrees(math.atan2(by - ay, bx - ax)) if (bx, by) != (ax, ay) else 0.0
    if angle > TURN_DEG:
        return "turning left"
    if angle < -TURN_DEG:
        return "turning right"
    if now >= 5 and end_speed < 0.5 * now:
        return "slowing"
    return "going straight"


def describe(example) -> str:
    word = TYPE_WORD.get(example["object_type"], str(example["object_type"]).capitalize())
    return f"{word} {movement(example['past_path'], example['true_future'])} · " \
           f"{speed_kmh(example['past_path']):.0f} km/h"
