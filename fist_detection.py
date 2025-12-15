import cv2
import mediapipe as mp
import math

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def debug_fist(hand_landmarks, frame_shape):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    wrist_px = (int(wrist.x * frame_shape[1]), int(wrist.y * frame_shape[0]))
    
    tips_indices = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    
    tips_px = [(int(hand_landmarks.landmark[i].x * frame_shape[1]),
                int(hand_landmarks.landmark[i].y * frame_shape[0])) for i in tips_indices]
    
    distances = [distance(tip, wrist_px) for tip in tips_px]
    
    print("Tip distances to wrist:", distances)
    
    return distances

# -------------------------------
# Mediapipe setup
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
draw_mp = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

print("🔹 Fist detection debug mode. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        distances = debug_fist(hand, frame.shape)
        
        # Optional: visualize landmarks
        draw_mp.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Fist Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
