#!/usr/bin/env python3
"""
MediaPipe "Magic Palm" Hand Tracking with Multi-Core Parallel Structure

Features:
- Threaded camera capture for reduced latency
- Multi-threaded Mediapipe inference utilizing all CPU cores
- Async drawing thread for smooth UI
- Exponential moving average landmark smoothing
- Handedness-aware finger detection
- Glowing fingertips and FPS overlay

Requirements: mediapipe, opencv-python, numpy, pyautogui
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import concurrent.futures
import queue
import multiprocessing
from collections import deque
from mediapipe.framework.formats import landmark_pb2

# Performance optimizations
cv2.setNumThreads(0)  # Let OpenCV use all cores

# ------- CONFIG -------
CAM_INDEX = 0
MODEL_COMPLEXITY = 1  # 1 for robust detection
MAX_NUM_HANDS = 2
MIN_DET_CONF = 0.5
MIN_TRACK_CONF = 0.5
SMOOTHING_ALPHA = 0.3  # Exponential moving average factor (0.1-0.5)
GLOW_COLOR = (0, 255, 255)
GLOW_RADIUS = 8
GLOW_THICKNESS = -1
DISPLAY_SCALE = 1.5
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
QUEUE_SIZE = 1  # Keep only latest frame
# ----------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Finger tip indices
FINGER_TIPS = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]

# Exponential Moving Average for landmark smoothing
class LandmarkSmoother:
    def __init__(self, alpha=SMOOTHING_ALPHA):
        self.alpha = alpha
        self.smoothed = None

    def update(self, landmarks):
        if self.smoothed is None:
            self.smoothed = [np.array([lm.x, lm.y, lm.z]) for lm in landmarks]
        else:
            for i, lm in enumerate(landmarks):
                current = np.array([lm.x, lm.y, lm.z])
                self.smoothed[i] = self.alpha * current + (1 - self.alpha) * self.smoothed[i]

        return [SimpleLandmark(x, y, z) for x, y, z in self.smoothed]

class SimpleLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# Convert normalized landmark to pixel coordinates
def lm_to_pixel(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

# Determine which fingers are up (True=extended)
def finger_states(hand_landmarks, handedness_label):
    """
    Detect which fingers are extended.
    Returns: list of booleans [thumb, index, middle, ring, pinky]
    """
    states = []

    # Thumb: adjust comparison for left vs right hand
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    if handedness_label == "Right":
        thumb_extended = thumb_tip.x < thumb_ip.x
    else:  # Left hand
        thumb_extended = thumb_tip.x > thumb_ip.x
    states.append(thumb_extended)

    # Fingers: tip y < pip y means extended (works with palm facing camera)
    for tip_idx, pip_idx in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
    ]:
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]
        states.append(tip.y < pip.y)

    return states

# Camera capture thread
def camera_thread(cap, frame_queue, stop_event):
    """Continuously capture frames and put latest into queue"""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            # Flip frame for natural mirroring
            frame = cv2.flip(frame, 1)
            # Keep only latest frame in queue
            try:
                frame_queue.put(frame, block=False)
            except queue.Full:
                try:
                    frame_queue.get_nowait()  # Remove old frame
                    frame_queue.put(frame, block=False)
                except queue.Empty:
                    pass
        time.sleep(0.001)  # Small delay to prevent busy waiting

# Mediapipe inference function
def process_frame(frame, hands, landmark_smoothers):
    """Process a single frame with Mediapipe hands"""
    h, w, _ = frame.shape

    # Convert to RGB for Mediapipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    hand_data = []

    if results.multi_hand_landmarks:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            if idx >= MAX_NUM_HANDS:
                break

            # Smooth landmarks
            smoothed_landmarks = landmark_smoothers[idx].update(hand_landmarks.landmark)

            # Create landmark list for drawing
            landmark_list = landmark_pb2.NormalizedLandmarkList()
            for lm in smoothed_landmarks:
                landmark_list.landmark.add(x=lm.x, y=lm.y, z=lm.z)

            # Get handedness
            label = handedness.classification[0].label

            # Finger states
            states = finger_states(landmark_list, label)
            fingers_up = sum(states)

            hand_data.append({
                'landmarks': landmark_list,
                'smoothed_landmarks': smoothed_landmarks,
                'label': label,
                'fingers_up': fingers_up,
                'idx': idx
            })

    return image, hand_data

# Drawing thread
def drawing_thread(display_queue, stop_event):
    """Draw landmarks and overlays on processed frames"""
    prev_time = 0

    while not stop_event.is_set():
        try:
            frame, hand_data = display_queue.get(timeout=0.1)
            h, w, _ = frame.shape

            hand_count = len(hand_data)

            for hand in hand_data:
                # Draw hand connections
                mp_drawing.draw_landmarks(
                    frame,
                    hand['landmarks'],
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Draw glowing fingertips
                for tip_idx in FINGER_TIPS:
                    lm = hand['smoothed_landmarks'][tip_idx]
                    x_px, y_px = lm_to_pixel(lm, w, h)
                    cv2.circle(frame, (x_px, y_px), GLOW_RADIUS, GLOW_COLOR, GLOW_THICKNESS)

                # Finger count overlay
                cv2.putText(frame, f"{hand['label']} {hand['fingers_up']}/5",
                           (10, 30 + 30*hand['idx']),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            if hand_count == 0:
                cv2.putText(frame, "No hands detected", (10, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Show scaled display
            cv2.imshow("Magic Palm Hands", cv2.resize(frame, (int(w*DISPLAY_SCALE), int(h*DISPLAY_SCALE))))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

        except queue.Empty:
            continue

def main():
    """Main function coordinating all threads"""
    # Setup camera
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"Error: Cannot open camera index {CAM_INDEX}")
        return

    # Setup Mediapipe with multi-threading
    num_threads = multiprocessing.cpu_count()
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRACK_CONF
    ) as hands:
        # Initialize landmark smoothers
        landmark_smoothers = [LandmarkSmoother() for _ in range(MAX_NUM_HANDS)]

        # Setup queues
        frame_queue = queue.Queue(maxsize=QUEUE_SIZE)
        display_queue = queue.Queue(maxsize=1)

        # Setup stop event
        stop_event = threading.Event()

        # Start camera thread
        camera_thread_handle = threading.Thread(
            target=camera_thread,
            args=(cap, frame_queue, stop_event),
            daemon=True
        )
        camera_thread_handle.start()

        # Start drawing thread
        drawing_thread_handle = threading.Thread(
            target=drawing_thread,
            args=(display_queue, stop_event),
            daemon=True
        )
        drawing_thread_handle.start()

        # Main processing loop with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            print(f"Magic Palm Hands started. Using {num_threads} CPU cores. Press 'q' to quit.")

            futures = {}

            while not stop_event.is_set():
                try:
                    # Get latest frame
                    frame = frame_queue.get(timeout=0.1)

                    # Submit inference task
                    future = executor.submit(process_frame, frame, hands, landmark_smoothers)
                    futures[future] = frame

                    # Check completed tasks
                    for future in list(futures.keys()):
                        if future.done():
                            try:
                                processed_frame, hand_data = future.result()
                                display_queue.put((processed_frame, hand_data), block=False)
                            except Exception as e:
                                print(f"Inference error: {e}")
                            del futures[future]

                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    stop_event.set()
                    break

        # Cleanup
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")

if __name__ == "__main__":
    main()
