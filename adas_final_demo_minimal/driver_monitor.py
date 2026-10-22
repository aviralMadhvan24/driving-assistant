
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import json
import os
from collections import deque
# ============================================================
# CONFIG
# ============================================================
CAMERA_INDEX = 0
DRIVER_STATE_FILE = "driver_state.json"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# Calibration
CALIBRATION_SECONDS = 3.0
# Head pose thresholds
YAW_ATTENTIVE = 18.0
YAW_DISTRACTED = 30.0
PITCH_ATTENTIVE = 15.0
PITCH_DISTRACTED = 25.0
# Head pose smoothing
SMOOTHING_SIZE = 10
# Frames required before declaring head distraction
DISTRACTION_FRAMES = 12
# ============================================================
# EYE / DROWSINESS CONFIG
# ============================================================
# EAR below this means eyes are probably closed.
EAR_CLOSED_THRESHOLD = 0.21
# EAR above this means eyes are open.
EAR_OPEN_THRESHOLD = 0.24
# How long eyes must stay closed before DROWSY
EYES_CLOSED_WARNING_TIME = 1.5
# Longer closure = severe drowsiness
EYES_CLOSED_CRITICAL_TIME = 3.0
# Blink duration upper limit
MAX_BLINK_DURATION = 0.8
# ============================================================
# MEDIAPIPE
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# ============================================================
# CAMERA
# ============================================================
# Use DirectShow on Windows, matching the successful webcam test.
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Allow the webcam to initialize and discard initial frames.
time.sleep(1)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    print("Try changing CAMERA_INDEX = 0 to CAMERA_INDEX = 1")
    raise SystemExit(1)

for _ in range(10):
    cap.grab()
# ============================================================
# 3D FACE MODEL
# ============================================================
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose
    (0.0, -63.6, -12.5),         # Chin
    (-43.3, 32.7, -26.0),        # Left eye
    (43.3, 32.7, -26.0),         # Right eye
    (-28.9, -28.9, -24.1),       # Left mouth
    (28.9, -28.9, -24.1)         # Right mouth
], dtype=np.float64)
# ============================================================
# EYE LANDMARKS
# ============================================================
# MediaPipe Face Mesh landmark indices
LEFT_EYE = [
    33,     # left corner
    160,    # upper
    158,    # upper
    133,    # right corner
    153,    # lower
    144     # lower
]
RIGHT_EYE = [
    362,    # left corner
    385,    # upper
    387,    # upper
    263,    # right corner
    373,    # lower
    380     # lower
]
# ============================================================
# SMOOTHING
# ============================================================
yaw_history = deque(maxlen=SMOOTHING_SIZE)
pitch_history = deque(maxlen=SMOOTHING_SIZE)
roll_history = deque(maxlen=SMOOTHING_SIZE)
ear_history = deque(maxlen=5)
# ============================================================
# CALIBRATION
# ============================================================
calibration_yaw = []
calibration_pitch = []
calibration_start = time.time()
calibrated = False
neutral_yaw = 0.0
neutral_pitch = 0.0
# ============================================================
# STATE
# ============================================================
distraction_counter = 0
eyes_closed = False
eyes_closed_start = None
blink_start = None
blink_count = 0
total_closed_time = 0.0
last_eye_state = "OPEN"
# ============================================================
# FUNCTIONS
# ============================================================
def get_image_points(landmarks, width, height):
    ids = [
        1,
        152,
        33,
        263,
        61,
        291
    ]
    points = []
    for idx in ids:
        lm = landmarks[idx]
        points.append([
            lm.x * width,
            lm.y * height
        ])
    return np.array(points, dtype=np.float64)
# ------------------------------------------------------------
# HEAD POSE
# ------------------------------------------------------------
def calculate_head_pose(image_points, width, height):
    focal_length = width
    center = (
        width / 2,
        height / 2
    )
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    distortion = np.zeros((4, 1))
    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]
    return pitch, yaw, roll
# ------------------------------------------------------------
# EYE ASPECT RATIO
# ------------------------------------------------------------
def calculate_ear(landmarks, eye_indices, width, height):
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append(
            np.array([
                lm.x * width,
                lm.y * height
            ])
        )
    # EAR:
    #
    #       p1 ----- p4
    #      /           \\
    #     p2           p6
    #     p3           p5
    #
    # EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|)
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]
    p5 = points[4]
    p6 = points[5]
    horizontal = np.linalg.norm(
        p1 - p4
    )
    vertical_1 = np.linalg.norm(
        p2 - p6
    )
    vertical_2 = np.linalg.norm(
        p3 - p5
    )
    if horizontal == 0:
        return 0.0
    ear = (
        vertical_1 + vertical_2
    ) / (
        2.0 * horizontal
    )
    return ear
# ------------------------------------------------------------
# ATTENTION
# ------------------------------------------------------------
def calculate_attention(yaw_error, pitch_error):
    yaw_score = max(
        0,
        1 - abs(yaw_error) / 45
    )
    pitch_score = max(
        0,
        1 - abs(pitch_error) / 35
    )
    score = (
        yaw_score *
        pitch_score *
        100
    )
    return max(
        0,
        min(100, score)
    )
# ------------------------------------------------------------
# HEAD DIRECTION
# ------------------------------------------------------------
def get_direction(yaw_error, pitch_error):
    if yaw_error > YAW_ATTENTIVE:
        return "RIGHT"
    if yaw_error < -YAW_ATTENTIVE:
        return "LEFT"
    if pitch_error > PITCH_ATTENTIVE:
        return "DOWN"
    if pitch_error < -PITCH_ATTENTIVE:
        return "UP"
    return "FORWARD"
# ------------------------------------------------------------
# TEXT
# ------------------------------------------------------------
def draw_text(
    frame,
    text,
    pos,
    scale=0.7,
    thickness=2
):
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )
# ============================================================
# ============================================================
# PUBLISH DRIVER STATE
# ============================================================
def publish_driver_state(driver_status, attention, drowsiness, head_status, eye_state, face_detected):
    state = {
        "driver_status": driver_status,
        "attention": float(attention),
        "drowsiness": float(drowsiness),
        "head_status": head_status,
        "eye_state": eye_state,
        "face_detected": bool(face_detected),
        "timestamp": time.time()
    }
    temp_file = DRIVER_STATE_FILE + ".tmp"
    try:
        with open(temp_file, "w") as f:
            json.dump(state, f)
        os.replace(temp_file, DRIVER_STATE_FILE)
    except Exception:
        pass
# MAIN
# ============================================================
previous_time = time.time()
print("=" * 60)
print("DRIVER MONITORING SYSTEM")
print("=" * 60)
print("Look straight at the camera for calibration.")
print("Press Q to quit.")
print("=" * 60)
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Camera frame failed; retrying...")
        time.sleep(0.2)
        continue
    # Mirror webcam
    frame = cv2.flip(
        frame,
        1
    )
    height, width = frame.shape[:2]
    # ========================================================
    # MEDIAPIPE
    # ========================================================
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )
    results = face_mesh.process(
        rgb
    )
    face_detected = False
    yaw = 0.0
    pitch = 0.0
    roll = 0.0
    yaw_error = 0.0
    pitch_error = 0.0
    attention = 0.0
    direction = "UNKNOWN"
    head_status = "INATTENTIVE"
    # Eye values
    left_ear = 0.0
    right_ear = 0.0
    avg_ear = 0.0
    eye_state = "UNKNOWN"
    drowsiness = 0.0
    # ========================================================
    # FACE DETECTED
    # ========================================================
    if results.multi_face_landmarks:
        face_detected = True
        face = results.multi_face_landmarks[0]
        # ----------------------------------------------------
        # DRAW FACE LANDMARKS
        # ----------------------------------------------------
        mp_drawing.draw_landmarks(
            frame,
            face,
            mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0),
                thickness=1,
                circle_radius=1
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 180, 0),
                thickness=1
            )
        )
        # ====================================================
        # HEAD POSE
        # ====================================================
        image_points = get_image_points(
            face.landmark,
            width,
            height
        )
        pose = calculate_head_pose(
            image_points,
            width,
            height
        )
        if pose is not None:
            current_pitch, current_yaw, current_roll = pose
            yaw_history.append(
                current_yaw
            )
            pitch_history.append(
                current_pitch
            )
            roll_history.append(
                current_roll
            )
            yaw = np.mean(
                yaw_history
            )
            pitch = np.mean(
                pitch_history
            )
            roll = np.mean(
                roll_history
            )
            # =================================================
            # CALIBRATION
            # =================================================
            if not calibrated:
                elapsed = (
                    time.time()
                    -
                    calibration_start
                )
                calibration_yaw.append(
                    yaw
                )
                calibration_pitch.append(
                    pitch
                )
                remaining = (
                    CALIBRATION_SECONDS
                    -
                    elapsed
                )
                cv2.putText(
                    frame,
                    f"CALIBRATING... {max(0, remaining):.1f}s",
                    (width // 2 - 150, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                if elapsed >= CALIBRATION_SECONDS:
                    neutral_yaw = np.median(
                        calibration_yaw
                    )
                    neutral_pitch = np.median(
                        calibration_pitch
                    )
                    calibrated = True
                    print()
                    print("Calibration complete.")
                    print(
                        f"Neutral yaw: {neutral_yaw:.2f}"
                    )
                    print(
                        f"Neutral pitch: {neutral_pitch:.2f}"
                    )
                    print()
            else:
                # =============================================
                # HEAD POSE ERROR
                # =============================================
                yaw_error = (
                    yaw -
                    neutral_yaw
                )
                pitch_error = (
                    pitch -
                    neutral_pitch
                )
                attention = calculate_attention(
                    yaw_error,
                    pitch_error
                )
                direction = get_direction(
                    yaw_error,
                    pitch_error
                )
                # =============================================
                # HEAD DISTRACTION
                # =============================================
                major_distraction = (
                    abs(yaw_error) >
                    YAW_DISTRACTED
                    or
                    abs(pitch_error) >
                    PITCH_DISTRACTED
                )
                if major_distraction:
                    distraction_counter += 1
                else:
                    distraction_counter = max(
                        0,
                        distraction_counter - 2
                    )
                if (
                    distraction_counter
                    >= DISTRACTION_FRAMES
                ):
                    head_status = "INATTENTIVE"
                elif (
                    abs(yaw_error) >
                    YAW_ATTENTIVE
                    or
                    abs(pitch_error) >
                    PITCH_ATTENTIVE
                ):
                    head_status = "DISTRACTED"
                else:
                    head_status = "ATTENTIVE"
        # ====================================================
        # EYE DETECTION
        # ====================================================
        left_ear = calculate_ear(
            face.landmark,
            LEFT_EYE,
            width,
            height
        )
        right_ear = calculate_ear(
            face.landmark,
            RIGHT_EYE,
            width,
            height
        )
        avg_ear = (
            left_ear +
            right_ear
        ) / 2.0
        ear_history.append(
            avg_ear
        )
        smoothed_ear = np.mean(
            ear_history
        )
        # ====================================================
        # EYE STATE
        # ====================================================
        if smoothed_ear < EAR_CLOSED_THRESHOLD:
            eye_state = "CLOSED"
        elif smoothed_ear > EAR_OPEN_THRESHOLD:
            eye_state = "OPEN"
        else:
            eye_state = last_eye_state
        # ====================================================
        # BLINK / EYE CLOSURE
        # ====================================================
        current_time = time.time()
        if (
            eye_state == "CLOSED"
            and last_eye_state == "OPEN"
        ):
            eyes_closed = True
            eyes_closed_start = current_time
            blink_start = current_time
        elif (
            eye_state == "OPEN"
            and last_eye_state == "CLOSED"
        ):
            if eyes_closed_start is not None:
                closure_duration = (
                    current_time
                    -
                    eyes_closed_start
                )
                total_closed_time += (
                    closure_duration
                )
                # Normal blink
                if (
                    closure_duration
                    <= MAX_BLINK_DURATION
                ):
                    blink_count += 1
            eyes_closed = False
            eyes_closed_start = None
        last_eye_state = eye_state
        # ====================================================
        # DROWSINESS
        # ====================================================
        if eyes_closed and eyes_closed_start is not None:
            closed_duration = (
                current_time
                -
                eyes_closed_start
            )
            if (
                closed_duration
                >= EYES_CLOSED_CRITICAL_TIME
            ):
                drowsiness = 100
            elif (
                closed_duration
                >= EYES_CLOSED_WARNING_TIME
            ):
                ratio = (
                    closed_duration
                    -
                    EYES_CLOSED_WARNING_TIME
                ) / (
                    EYES_CLOSED_CRITICAL_TIME
                    -
                    EYES_CLOSED_WARNING_TIME
                )
                drowsiness = (
                    70 +
                    ratio * 30
                )
            else:
                drowsiness = (
                    closed_duration
                    /
                    EYES_CLOSED_WARNING_TIME
                ) * 50
        else:
            # Gradually recover after opening eyes
            drowsiness *= 0.92
    else:
        # ====================================================
        # NO FACE
        # ====================================================
        yaw_history.clear()
        pitch_history.clear()
        roll_history.clear()
        ear_history.clear()
        distraction_counter += 1
        head_status = "INATTENTIVE"
        eye_state = "NO FACE"
        drowsiness = 100
    # ========================================================
    # DRIVER STATUS
    # ========================================================
    if not face_detected:
        driver_status = "INATTENTIVE"
    elif drowsiness >= 70:
        driver_status = "DROWSY"
    elif head_status == "INATTENTIVE":
        driver_status = "INATTENTIVE"
    elif head_status == "DISTRACTED":
        driver_status = "DISTRACTED"
    else:
        driver_status = "ATTENTIVE"
    publish_driver_state(
        driver_status, attention, drowsiness,
        head_status, eye_state, face_detected
    )
    # ========================================================
    # HUD
    # ========================================================
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (425, 365),
        (0, 0, 0),
        -1
    )
    frame = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )
    draw_text(
        frame,
        "DRIVER MONITORING",
        (25, 42),
        0.8,
        2
    )
    if face_detected:
        draw_text(
            frame,
            "Face: DETECTED",
            (25, 75)
        )
        if calibrated:
            draw_text(
                frame,
                f"Head: {direction}",
                (25, 105)
            )
            draw_text(
                frame,
                f"Yaw: {yaw_error:.1f} deg",
                (25, 135)
            )
            draw_text(
                frame,
                f"Pitch: {pitch_error:.1f} deg",
                (25, 165)
            )
            draw_text(
                frame,
                f"Attention: {attention:.0f}%",
                (25, 195)
            )
        else:
            draw_text(
                frame,
                "Calibration: RUNNING",
                (25, 105)
            )
        # ====================================================
        # EYE HUD
        # ====================================================
        draw_text(
            frame,
            f"Eyes: {eye_state}",
            (25, 225)
        )
        draw_text(
            frame,
            f"EAR: {avg_ear:.3f}",
            (25, 255)
        )
        draw_text(
            frame,
            f"Drowsiness: {drowsiness:.0f}%",
            (25, 285)
        )
        draw_text(
            frame,
            f"Blinks: {blink_count}",
            (25, 315)
        )
        draw_text(
            frame,
            f"Status: {driver_status}",
            (25, 345)
        )
    else:
        draw_text(
            frame,
            "Face: NOT DETECTED",
            (25, 85)
        )
        draw_text(
            frame,
            "Eyes: UNKNOWN",
            (25, 120)
        )
        draw_text(
            frame,
            "Drowsiness: 100%",
            (25, 155)
        )
        draw_text(
            frame,
            "Status: INATTENTIVE",
            (25, 190)
        )
    # ========================================================
    # WARNINGS
    # ========================================================
    if driver_status == "DROWSY":
        cv2.rectangle(
            frame,
            (0, 0),
            (width - 1, height - 1),
            (0, 0, 255),
            6
        )
        cv2.putText(
            frame,
            "DROWSINESS WARNING",
            (width // 2 - 170, height - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "WAKE UP / TAKE A BREAK",
            (width // 2 - 180, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
    elif driver_status == "INATTENTIVE":
        cv2.rectangle(
            frame,
            (0, 0),
            (width - 1, height - 1),
            (0, 0, 255),
            5
        )
        cv2.putText(
            frame,
            "DRIVER ATTENTION WARNING",
            (width // 2 - 190, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
    elif driver_status == "DISTRACTED":
        cv2.putText(
            frame,
            "PLEASE LOOK AT THE ROAD",
            (width // 2 - 170, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
            cv2.LINE_AA
        )
    # ========================================================
    # FPS
    # ========================================================
    now = time.time()
    dt = now - previous_time
    fps = (
        1.0 / dt
        if dt > 0
        else 0
    )
    previous_time = now
    draw_text(
        frame,
        f"FPS: {fps:.1f}",
        (width - 110, 30),
        0.55,
        2
    )
    # ========================================================
    # DISPLAY
    # ========================================================
    cv2.imshow(
        "Driver Monitoring System",
        frame
    )
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
# ============================================================
# CLEANUP
# ============================================================
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("Driver monitoring stopped.")
