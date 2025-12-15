import cv2
import mediapipe as mp
import math
import time
from pynput.keyboard import Controller, Key

# ────────────────────────────────
# Keyboard controller
# ────────────────────────────────
keyboard = Controller()
def press_key_for_gesture(direction):
    keymap = {
        "UP": Key.up,
        "DOWN": Key.down,
        "LEFT": Key.left,
        "RIGHT": Key.right,
        "ENTER": Key.enter
    }
    key = keymap.get(direction)
    if key:
        keyboard.press(key)
        keyboard.release(key)

# ────────────────────────────────
# Constants
# ────────────────────────────────
CENTER_PADDING_RADIUS = 15
MIN_ZONE_RADIUS = 15
MAX_ZONE_RADIUS = 60
MIN_TRIGGER_DISTANCE = 10
EMA_ALPHA = 0.4
SLOW_FOLLOW_SPEED = 0.1
FAST_FOLLOW_SPEED = 0.2
GESTURE_RECENTER_DELAY = 0.4
ACTION_CIRCLE_RADIUS = 40
DEBUG = True

# ────────────────────────────────
# State
# ────────────────────────────────
center_point = None
entry_point = None
current_zone = "CENTER"
momentary_gesture_active = False
last_momentary_gesture_time = 0
prev_filtered_pos = None
prev_radius_x = MAX_ZONE_RADIUS
prev_radius_y = MAX_ZONE_RADIUS
fist_active = False

# Action Circle State
action_circle_active = False
action_circle_pos = None

# ────────────────────────────────
# Helpers
# ────────────────────────────────
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def get_zone(pos, center, radius_x, radius_y):
    dx = pos[0] - center[0]
    dy = center[1] - pos[1]  # invert Y axis
    if (dx*dx)/(radius_x*radius_x) + (dy*dy)/(radius_y*radius_y) < 1 or math.hypot(dx, dy) < CENTER_PADDING_RADIUS:
        return "CENTER"
    angle = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    if 45 <= angle < 135:
        return "RIGHT"
    elif 135 <= angle < 225:
        return "DOWN"
    elif 225 <= angle < 315:
        return "LEFT"
    else:
        return "UP"

def draw_ui(frame, center, pos, zone, radius_x, radius_y):
    cv2.ellipse(frame, center, (int(radius_x), int(radius_y)), 0, 0, 360, (90,90,90), 2)
    cv2.circle(frame, center, CENTER_PADDING_RADIUS, (60,60,60), 1)
    cv2.circle(frame, pos, 8, (0,255,0), -1)
    cv2.putText(frame, f"ZONE: {zone}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

def draw_action_circle(frame, pos, radius):
    cv2.circle(frame, pos, radius, (0,165,255), 3)
    cv2.putText(frame, "ACTION", (pos[0]-radius//2, pos[1]+radius+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

def is_fist(hand_landmarks, frame_shape, DEBUG=False):
    """
    Detects fist with individual finger margins.
    Each finger has its own threshold ratio to account for finger length differences.
    """
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    wrist_px = (wrist.x * frame_shape[1], wrist.y * frame_shape[0])

    # Fingers: (tip, mcp, threshold ratio)
    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP, 0.45),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP, 0.50),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP, 0.55),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP, 0.50)
    ]

    # Reference hand size (wrist to middle MCP)
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    middle_mcp_px = (middle_mcp.x * frame_shape[1], middle_mcp.y * frame_shape[0])
    hand_size = distance(wrist_px, middle_mcp_px)

    curled = []
    for tip_idx, mcp_idx, threshold in fingers:
        tip = hand_landmarks.landmark[tip_idx]
        mcp = hand_landmarks.landmark[mcp_idx]
        tip_px = (tip.x * frame_shape[1], tip.y * frame_shape[0])
        mcp_px = (mcp.x * frame_shape[1], mcp.y * frame_shape[0])
        ratio = distance(tip_px, mcp_px) / hand_size
        curled.append((ratio, threshold))

    if DEBUG:
        print("Finger ratios vs thresholds:", [(round(r,2), t) for r, t in curled])

    # All fingers must satisfy their individual thresholds
    return all(r <= t for r, t in curled)

# ────────────────────────────────
# Mediapipe setup
# ────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
draw_mp = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

print("🎯 Radial gestures + dynamic oval + action circle with reliable fist")
print("Press Q to quit")

# ────────────────────────────────
# Main loop
# ────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    current_time = time.time()

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        raw_pos = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
        if prev_filtered_pos is None:
            filtered_pos = raw_pos
        else:
            filtered_pos = (
                int(EMA_ALPHA * raw_pos[0] + (1 - EMA_ALPHA) * prev_filtered_pos[0]),
                int(EMA_ALPHA * raw_pos[1] + (1 - EMA_ALPHA) * prev_filtered_pos[1])
            )
        prev_filtered_pos = filtered_pos

        # Initialize center
        if center_point is None:
            center_point = filtered_pos
            entry_point = filtered_pos

        dx = filtered_pos[0] - center_point[0]
        dy = filtered_pos[1] - center_point[1]

        # Dynamic oval radius
        if momentary_gesture_active or action_circle_active:
            radius_x = radius_y = MAX_ZONE_RADIUS
        else:
            radius_x = MAX_ZONE_RADIUS - abs(dx)
            radius_y = MAX_ZONE_RADIUS - abs(dy)
            radius_x = max(MIN_ZONE_RADIUS, radius_x)
            radius_y = max(MIN_ZONE_RADIUS, radius_y)

        radius_x = int(EMA_ALPHA * radius_x + (1 - EMA_ALPHA) * prev_radius_x)
        radius_y = int(EMA_ALPHA * radius_y + (1 - EMA_ALPHA) * prev_radius_y)
        prev_radius_x = radius_x
        prev_radius_y = radius_y

        # --- Action Circle Mode ---
        if action_circle_active:
            if action_circle_pos is None:
                action_circle_pos = filtered_pos
            else:
                action_circle_pos = (
                    int(action_circle_pos[0] + (filtered_pos[0]-action_circle_pos[0])*SLOW_FOLLOW_SPEED),
                    int(action_circle_pos[1] + (filtered_pos[1]-action_circle_pos[1])*SLOW_FOLLOW_SPEED)
                )
            if distance(filtered_pos, action_circle_pos) < ACTION_CIRCLE_RADIUS:
                press_key_for_gesture("ENTER")
                action_circle_active = False
                if DEBUG: print("ACTION CIRCLE ENTER TRIGGERED")
            draw_action_circle(frame, action_circle_pos, ACTION_CIRCLE_RADIUS)

        # --- Radial gestures ---
        elif not action_circle_active:
            new_zone = get_zone(filtered_pos, center_point, radius_x, radius_y)
            if new_zone != current_zone:
                current_zone = new_zone
                entry_point = filtered_pos
                if new_zone != "CENTER":
                    momentary_gesture_active = False

            if current_zone != "CENTER" and not momentary_gesture_active:
                if distance(filtered_pos, entry_point) >= MIN_TRIGGER_DISTANCE:
                    press_key_for_gesture(current_zone)
                    momentary_gesture_active = True
                    last_momentary_gesture_time = current_time
                    if DEBUG: print("GESTURE:", current_zone)

            if momentary_gesture_active and current_time - last_momentary_gesture_time >= GESTURE_RECENTER_DELAY:
                center_point = filtered_pos
                entry_point = filtered_pos
                momentary_gesture_active = False
                if DEBUG: print("Auto-recenter after gesture")

        # --- Fist detection (second hand triggers action circle) ---
        if len(result.multi_hand_landmarks) > 1:
            hand2 = result.multi_hand_landmarks[1]
            if is_fist(hand2, frame.shape) and not fist_active:
                action_circle_active = True
                action_circle_pos = None
                fist_active = True
                if DEBUG: print("ACTION CIRCLE ACTIVATED")
            elif not is_fist(hand2, frame.shape) and fist_active:
                fist_active = False
                if DEBUG: print("FIST RELEASED")
            draw_mp.draw_landmarks(frame, hand2, mp_hands.HAND_CONNECTIONS)

        draw_mp.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        draw_ui(frame, center_point, filtered_pos, current_zone, radius_x, radius_y)

    elif fist_active:
        fist_active = False
        if DEBUG: print("FIST RELEASED (No hands detected)")

    cv2.imshow("Radial Gesture Wheel", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ────────────────────────────────
# Cleanup
# ────────────────────────────────
cap.release()
cv2.destroyAllWindows()
hands.close()
print("✓ Clean shutdown")
