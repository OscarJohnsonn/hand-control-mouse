import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import numpy as np

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize keyboard controller
keyboard = Controller()

# Flags for key press status
keys_pressed = {'w': False, 'a': False, 's': False, 'd': False}

# Function to simulate key press
def press_key(key, press=True):
    if press and not keys_pressed[key]:
        keyboard.press(key)
        keys_pressed[key] = True
    elif not press and keys_pressed[key]:
        keyboard.release(key)
        keys_pressed[key] = False

# Function to check if the hand is fully extended or closed
def is_hand_open(hand_landmarks):
    # Check the distance between the fingertips and the palm
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # A hand is considered open if the tips are far enough from the wrist
    open_threshold = 0.1  # Threshold for determining if the hand is open

    distances = [
        np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([wrist.x, wrist.y])),
        np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([wrist.x, wrist.y])),
        np.linalg.norm(np.array([middle_tip.x, middle_tip.y]) - np.array([wrist.x, wrist.y])),
        np.linalg.norm(np.array([ring_tip.x, ring_tip.y]) - np.array([wrist.x, wrist.y])),
        np.linalg.norm(np.array([pinky_tip.x, pinky_tip.y]) - np.array([wrist.x, wrist.y])),
    ]
    
    return all(dist > open_threshold for dist in distances)

# Start capturing video
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = hands.process(image)

        # Reset key presses
        for key in keys_pressed:
            press_key(key, press=False)

        # Track current key press
        current_key = "None"

        # If hands are detected
        if results.multi_hand_landmarks:
            left_hand_open = right_hand_open = False
            for hand_landmarks in results.multi_hand_landmarks:
                # Check if the hand is open
                is_open = is_hand_open(hand_landmarks)
                
                # Identify left or right hand based on wrist position (x coordinate)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                if wrist.x < 0.5:  # Left hand (closer to the left of the screen)
                    left_hand_open = is_open
                else:  # Right hand (closer to the right of the screen)
                    right_hand_open = is_open

                # Draw landmarks on the hand
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Control logic based on hand states
            if left_hand_open and right_hand_open:
                press_key('w', press=True)
                current_key = "w"
            elif not left_hand_open and not right_hand_open:
                press_key('s', press=True)
                current_key = "s"
            elif right_hand_open:
                press_key('a', press=True)
                current_key = "a"
            elif left_hand_open:
                press_key('d', press=True)
                current_key = "d"

        # Display the current key pressed
        cv2.putText(frame, f"Current Key: {current_key}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the video feed
        cv2.imshow('Hand Tracking', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Release any pressed keys
    for key in keys_pressed:
        press_key(key, press=False)
