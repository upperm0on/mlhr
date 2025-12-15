from pynput import keyboard

# Create a single controller instance ONCE
controller = keyboard.Controller()

# Map gestures to keyboard keys
GESTURE_KEY_MAP = {
    'UP': keyboard.Key.up,
    'DOWN': keyboard.Key.down,
    'LEFT': keyboard.Key.left,
    'RIGHT': keyboard.Key.right,
}

def press_key_for_gesture(gesture: str):
    """
    Presses and releases a keyboard key corresponding to a recognized gesture.
    """
    key_to_press = GESTURE_KEY_MAP.get(gesture)
    if key_to_press:
        controller.press(key_to_press)
        controller.release(key_to_press)
