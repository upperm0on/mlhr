# ML Hand Gesture Control for Keyboard and Mouse

This project enables intuitive control of your computer's keyboard and mouse through hand gestures detected via a webcam, leveraging MediaPipe for hand tracking and pynput for input simulation.

## Features

- **Radial Gestures**: Control directional inputs (UP, DOWN, LEFT, RIGHT) by moving your index finger within a dynamic radial zone.
- **Index-Thumb Touch**: Tap your index finger and thumb together to simulate a single `ENTER` key press.
- **Fist Detection for Action Circle**: Use a second hand to form a fist. This activates an "Action Circle" on screen. Move your primary hand (index finger) into this circle to trigger a single `ENTER` key press.
- **Dynamic Zone Radius**: The radial gesture zones dynamically adjust based on hand movement, providing a more fluid control experience.
- **Slap Detection**: Quickly bringing your index and middle fingers together can trigger `ESC` (to exit focus mode) or `TAB` (to enter focus mode), enabling quick context switching.
- **Visual Feedback**: Real-time visualization of hand landmarks, gesture zones, and action circles directly on the webcam feed.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/mlhr.git
    cd mlhr
    ```
2.  **Create a virtual environment** (recommended):
    ```bash
    python3 -m venv mlhr_env
    source mlhr_env/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install opencv-python mediapipe pynput
    ```

## Usage

1.  **Run the script**:
    ```bash
    python hand_tracking.py
    ```
2.  **Webcam Feed**: A window will open displaying your webcam feed with hand tracking overlays.
3.  **Quit**: Press `Q` to exit the application.

## Gesture Guide

-   **Radial Navigation (Primary Hand - Index Finger)**:
    -   Move your index finger up, down, left, or right relative to the dynamic center point to trigger corresponding arrow keys.
-   **Enter (Primary Hand - Index Finger & Thumb)**:
    -   Touch your index finger tip to your thumb tip to send a single `ENTER` key press.
-   **Action Circle (Secondary Hand - Fist, then Primary Hand)**:
    -   Form a **fist** with your **second hand** to activate an orange "ACTION" circle on screen.
    -   Move your **primary hand's index finger** into this orange circle to trigger a single `ENTER` key press.
-   **Focus Mode Toggle (Primary Hand - Index & Middle Finger "Slap")**:
    -   Quickly bring your index and middle fingers together (a "slap" gesture) with your primary hand:
        -   First slap: Sends `ESC` and turns "FOCUS: OFF".
        -   Second consecutive slap within a time window: Sends `TAB` and turns "FOCUS: ON".

Enjoy controlling your computer with gestures!
