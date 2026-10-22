import argparse
import csv
import math
import os
import time
import traceback
from collections import deque

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Tank coordinates use the same corner order as cam-recorder-original.py:
# origin -> +x corner -> opposite corner -> +y corner.
TANK_WIDTH_MM = 760.0
TANK_HEIGHT_MM = 760.0
GRID_SPACING_MM = 50.0

STRENGTH = 1.0
THRESH = 90
USE_OTSU = True
MIN_AREA = 500
MAX_AREA_PCT = 20
MIN_CIRC = 0.55
BLUR = 4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(SCRIPT_DIR, "snapshots")
WIN_NAME = "Undistorted + Sphero Tracking"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track a Sphero and move it to a tank coordinate in millimeters."
    )
    parser.add_argument("--target-x", type=float, help="Target x coordinate in tank mm")
    parser.add_argument("--target-y", type=float, help="Target y coordinate in tank mm")
    parser.add_argument("--speed", type=int, default=100, help="Sphero roll speed, 1-255")
    # set default = 60 for rolling on ground
    parser.add_argument(
        "--speed-mm-s",
        type=float,
        default=118.0,
        # set default = 470.0 for rolling on ground --> A speed setting of 60 is roughly 470 mm/s if you scale proportionally from BOLT’s listed 2 m/s top speed: \(60/255 \times 2000 \approx 471\)
        help="Estimated travel speed used to calculate roll duration",
    )
    args = parser.parse_args()
    if (args.target_x is None) != (args.target_y is None):
        parser.error("--target-x and --target-y must be provided together")
    if not 1 <= args.speed <= 255:
        parser.error("--speed must be between 1 and 255")
    if args.speed_mm_s <= 0:
        parser.error("--speed-mm-s must be positive")
    return args


def find_ball(gray):
    """Return (center_x, center_y, radius) for the best dark round blob."""
    processed = gray
    if BLUR:
        kernel_size = 2 * BLUR + 1
        processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)

    if USE_OTSU:
        _, mask = cv2.threshold(
            processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, mask = cv2.threshold(processed, THRESH, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    height, width = gray.shape
    max_area = height * width * MAX_AREA_PCT / 100.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > max_area:
            continue
        (_, _), radius = cv2.minEnclosingCircle(contour)
        if radius <= 0:
            continue
        circularity = area / (np.pi * radius * radius)
        if circularity < MIN_CIRC:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]

        contour_mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        darkness = 1.0 - float(gray[contour_mask > 0].mean()) / 255.0
        score = darkness * circularity
        if score > best_score:
            best_score = score
            best = (center_x, center_y, radius)

    return best, mask


def img_to_tank(points_px, homography):
    points = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def draw_grid_overlay(view, inverse_homography):
    for x in np.arange(0, TANK_WIDTH_MM + 1e-6, GRID_SPACING_MM):
        tank_points = np.array(
            [[x, 0], [x, TANK_HEIGHT_MM]], dtype=np.float32
        ).reshape(-1, 1, 2)
        image_points = cv2.perspectiveTransform(
            tank_points, inverse_homography
        ).reshape(-1, 2)
        cv2.line(
            view,
            tuple(image_points[0].astype(int)),
            tuple(image_points[1].astype(int)),
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

    for y in np.arange(0, TANK_HEIGHT_MM + 1e-6, GRID_SPACING_MM):
        tank_points = np.array(
            [[0, y], [TANK_WIDTH_MM, y]], dtype=np.float32
        ).reshape(-1, 1, 2)
        image_points = cv2.perspectiveTransform(
            tank_points, inverse_homography
        ).reshape(-1, 2)
        cv2.line(
            view,
            tuple(image_points[0].astype(int)),
            tuple(image_points[1].astype(int)),
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )


def save_tank_analysis(trail_px, homography):
    if len(trail_px) < 2:
        print("Not enough tracked points to save a trajectory.")
        return

    points = img_to_tank(list(trail_px), homography)
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.set_xlim(0, TANK_WIDTH_MM)
    axis.set_ylim(TANK_HEIGHT_MM, 0)
    axis.set_aspect("equal")
    axis.set_xticks(np.arange(0, TANK_WIDTH_MM + 1e-6, GRID_SPACING_MM))
    axis.set_yticks(np.arange(0, TANK_HEIGHT_MM + 1e-6, GRID_SPACING_MM))
    axis.grid(True, color="blue", linewidth=0.6)
    axis.plot(points[:, 0], points[:, 1], "-o", color="orange", markersize=3)
    axis.plot(points[-1, 0], points[-1, 1], "r*", markersize=14)
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_title(f"Tank trajectory ({TANK_WIDTH_MM:.0f} x {TANK_HEIGHT_MM:.0f} mm)")

    timestamp = int(time.time())
    image_path = os.path.join(SNAPSHOT_DIR, f"tank_grid_{timestamp}.png")
    csv_path = os.path.join(SNAPSHOT_DIR, f"tank_grid_{timestamp}.csv")
    figure.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    with open(csv_path, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["x_mm", "y_mm"])
        writer.writerows(points.tolist())
    print(f"Saved grid plot to: {image_path}")
    print(f"Saved tank points to: {csv_path}")


def move_sphero_to_coordinate(
    current_x_mm,
    current_y_mm,
    target_x_mm,
    target_y_mm,
    speed,
    speed_mm_per_second,
):
    """Move from the measured position to a target using SpheroEduAPI roll."""
    if not 0 <= target_x_mm <= TANK_WIDTH_MM:
        raise ValueError(f"target x must be between 0 and {TANK_WIDTH_MM:.0f} mm")
    if not 0 <= target_y_mm <= TANK_HEIGHT_MM:
        raise ValueError(f"target y must be between 0 and {TANK_HEIGHT_MM:.0f} mm")

    delta_x = target_x_mm - current_x_mm
    delta_y = target_y_mm - current_y_mm
    distance_mm = math.hypot(delta_x, delta_y)
    if distance_mm < 1.0:
        print("Sphero is already within 1 mm of the target.")
        return

    # Sphero heading 0 is forward. Tank y increases downward, so -dy is forward.
    # Negate the angle because positive tank x points left in the robot's view.
    heading = -math.degrees(math.atan2(delta_x, -delta_y))
    command_heading = heading % 360
    duration = distance_mm / speed_mm_per_second # TO-DO: ASSUMES THIS IS 200 MM/S, MUST FIND SOME CONSTANT TO TRANSLATE SPEED TO LINEAR VELOCITY

    try:
        from spherov2 import scanner
        from spherov2.sphero_edu import SpheroEduAPI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "spherov2 is required for movement. Install it with "
            "'python -m pip install spherov2' in the active environment."
        ) from exc

    print(
        f"Moving {distance_mm:.1f} mm at heading {heading:.1f} degrees "
        f"for {duration:.2f} seconds."
    )
    last_error = None
    for attempt in range(1, 4):
        try:
            toy = scanner.find_toy(timeout=8.0)
            with SpheroEduAPI(toy) as droid:
                print(
                    f"Connected to Sphero: {toy.name} ({toy.address})",
                    flush=True,
                )
                #droid.set_rgb_led(255, 0, 0)
                droid.reset_aim()
                print(
                    f"Executing roll: heading={heading:.1f} degrees "
                    f"({command_heading:.1f} degrees sent), speed={int(speed)}, "
                    f"duration={float(duration):.2f}s",
                    flush=True,
                )
                print(
                    f"Current Sphero coordinates: ({current_x_mm:.1f}, "
                    f"{current_y_mm:.1f}) mm",
                    flush=True,
                )
                droid.roll(int(command_heading), int(speed), float(duration))
            print(
                f"Move command sent successfully: target=({target_x_mm:.1f}, "
                f"{target_y_mm:.1f}) mm, heading={heading:.1f} degrees, "
                f"duration={duration:.2f} seconds.",
                flush=True,
            )
            return
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                print(
                    f"Sphero connection attempt {attempt} failed: "
                    f"{type(exc).__name__}: {exc!r}. Rescanning...",
                    flush=True,
                )
                time.sleep(1.0)

    raise RuntimeError(
        "Could not connect to the Sphero after 3 attempts. Wake the Sphero, "
        "keep it close to the Mac, and check macOS Bluetooth permissions. "
        f"Last error: {type(last_error).__name__}: {last_error!r}"
    ) from last_error


def main():
    args = parse_args()
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    target_coordinate = None
    if args.target_x is not None:
        target_coordinate = (args.target_x, args.target_y)
        print(
            f"Target queued: ({args.target_x:.1f}, {args.target_y:.1f}) mm. "
            "Calibrate the tank with 'p'."
        )

    capture = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not capture.isOpened():
        raise RuntimeError("Could not open camera. Check camera index and macOS permissions.")

    corner_points = []
    homography = None
    inverse_homography = None
    picking_corners = False
    move_attempted = False
    trail = deque(maxlen=150)
    current_tank_coordinate = None

    def on_mouse(event, x, y, flags, param):
        nonlocal picking_corners, corner_points, homography, inverse_homography
        if event != cv2.EVENT_LBUTTONDOWN or not picking_corners:
            return
        corner_points.append((x, y))
        print(f"Corner {len(corner_points)}/4 picked: ({x}, {y})")
        if len(corner_points) == 4:
            source = np.float32(corner_points)
            destination = np.float32(
                [
                    [0, 0],
                    [TANK_WIDTH_MM, 0],
                    [TANK_WIDTH_MM, TANK_HEIGHT_MM],
                    [0, TANK_HEIGHT_MM],
                ]
            )
            homography = cv2.getPerspectiveTransform(source, destination)
            inverse_homography = cv2.getPerspectiveTransform(destination, source)
            picking_corners = False
            print("Tank calibration complete.")

    cv2.namedWindow(WIN_NAME)
    cv2.setMouseCallback(WIN_NAME, on_mouse)

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                print("ERROR: Camera failed to read a frame.")
                break

            height, width = frame.shape[:2]
            camera_matrix = np.array(
                [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
                dtype=np.float32,
            )
            distortion = np.array(
                [-STRENGTH, STRENGTH * 0.3, 0, 0], dtype=np.float32
            )
            undistorted = cv2.undistort(frame, camera_matrix, distortion)
            hit, mask = find_ball(cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY))
            view = undistorted.copy()

            if inverse_homography is not None:
                draw_grid_overlay(view, inverse_homography)

            if hit:
                center_x, center_y, radius = hit
                trail.append((int(center_x), int(center_y)))
                if homography is not None:
                    current_tank_coordinate = img_to_tank(
                        [(center_x, center_y)], homography
                    )[0]
                cv2.circle(view, (int(center_x), int(center_y)), int(radius), (0, 255, 0), 2)
                cv2.drawMarker(
                    view,
                    (int(center_x), int(center_y)),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    20,
                    2,
                )

                if target_coordinate is not None and homography is not None and not move_attempted:
                    move_attempted = True
                    current_x, current_y = img_to_tank(
                        [(center_x, center_y)], homography
                    )[0]
                    try:
                        move_sphero_to_coordinate(
                            current_x,
                            current_y,
                            target_coordinate[0],
                            target_coordinate[1],
                            args.speed,
                            args.speed_mm_s,
                        )
                    except Exception as exc:
                        print(
                            f"Move failed: {type(exc).__name__}: {exc!r}",
                            flush=True,
                        )
                        traceback.print_exc()

                cv2.putText(
                    view,
                    f"pixel ({center_x:.0f}, {center_y:.0f})",
                    (int(center_x) + 10, int(center_y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    view,
                    "NO BALL FOUND",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            for index in range(1, len(trail)):
                cv2.line(view, trail[index - 1], trail[index], (0, 180, 255), 2)

            if picking_corners:
                for index, point in enumerate(corner_points):
                    cv2.circle(view, point, 5, (0, 255, 255), -1)
                    cv2.putText(
                        view,
                        str(index + 1),
                        (point[0] + 8, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )
                cv2.putText(
                    view,
                    f"Click corner {len(corner_points) + 1}/4: origin, +x, opposite, +y",
                    (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            if target_coordinate is not None and not move_attempted:
                target_status = (
                    f"target ({target_coordinate[0]:.0f}, {target_coordinate[1]:.0f}) mm"
                )
            elif move_attempted:
                target_status = "move command sent"
            else:
                target_status = "no target; use --target-x/--target-y or press m"
            cv2.putText(
                view,
                target_status,
                (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Original", frame)
            cv2.imshow(WIN_NAME, view)
            cv2.imshow("Mask", mask)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("p"):
                picking_corners = True
                corner_points = []
                homography = None
                inverse_homography = None
                current_tank_coordinate = None
                move_attempted = False
                print("Click the tank corners in order: origin, +x, opposite, +y.")
            elif key == ord("r"):
                picking_corners = False
                corner_points = []
                homography = None
                inverse_homography = None
                current_tank_coordinate = None
                move_attempted = False
                print("Tank calibration reset.")
            elif key == ord("m"):
                try:
                    target_coordinate = (
                        float(input("Target x in mm: ")),
                        float(input("Target y in mm: ")),
                    )
                    move_attempted = False
                    print(f"Target queued: {target_coordinate} mm")
                except ValueError:
                    print("Target must contain two numeric coordinates.")
            elif key == ord("c"):
                if current_tank_coordinate is None:
                    print("Current position unavailable. Calibrate with 'p' and show the Sphero.")
                else:
                    print(
                        f"Current Sphero coordinates: "
                        f"({current_tank_coordinate[0]:.1f}, "
                        f"{current_tank_coordinate[1]:.1f}) mm",
                        flush=True,
                    )
            elif key == ord("s"):
                image_path = os.path.join(SNAPSHOT_DIR, f"tracked_{int(time.time())}.png")
                cv2.imwrite(image_path, view)
                print(f"Saved to: {image_path}")
                if homography is not None:
                    save_tank_analysis(trail, homography)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
