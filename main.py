import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Reduce frame size for better performance
frame_width = 320  # Reduced resolution
frame_height = 240  # Reduced resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Throttle mouse movement updates
mouse_update_threshold = 1  # Pixels

# Limit the frame rate
fps = 100  # Target frame rate
frame_delay = 0.0 / fps

# Disable PyAutoGUI failsafe
pyautogui.FAILSAFE = False

# Low-pass filter parameters
alpha = 1.0  # Smoothing factor

# Scrolling sensitivity factor
scroll_sensitivity = 0.1  # Reduce this value to make scrolling less sensitive

# Helper functions
def calculate_distance(point1, point2):
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

def is_hand_open(hand_landmarks):
    # Calculate distances between the tips of the fingers and the base of the palm
    base_palm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    distances = [
        calculate_distance(base_palm, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]),
        calculate_distance(base_palm, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]),
        calculate_distance(base_palm, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]),
        calculate_distance(base_palm, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP])
    ]
    # If all distances are above a certain threshold, the hand is open
    return all(distance > 0.2 for distance in distances)

def process_frame(frame, last_mouse_position, smoothed_position, last_scroll_position, smoothed_scroll_position):
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hands
    hand_result = hands.process(rgb_frame)

    # Hand Tracking: Control cursor and clicking
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            if is_hand_open(hand_landmarks):
                # Use the hand's vertical movement to scroll
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                screen_y = int(index_finger_tip.y * screen_height)
                scroll_amount = (screen_y - last_scroll_position) * scroll_sensitivity

                # Apply low-pass filter to smooth the scrolling
                smoothed_scroll_position = alpha * scroll_amount + (1 - alpha) * smoothed_scroll_position
                pyautogui.scroll(-smoothed_scroll_position)
                last_scroll_position = screen_y
            else:
                # Get the coordinates of the index finger tip and thumb tip
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Map the coordinates to the screen size
                screen_x = int(index_finger_tip.x * screen_width)
                screen_y = int(index_finger_tip.y * screen_height)

                # Apply low-pass filter to smooth the cursor movement
                smoothed_position = (
                    alpha * screen_x + (1 - alpha) * smoothed_position[0],
                    alpha * screen_y + (1 - alpha) * smoothed_position[1]
                )

                # Move the mouse cursor if position has changed significantly
                if abs(last_mouse_position[0] - smoothed_position[0]) > mouse_update_threshold or \
                   abs(last_mouse_position[1] - smoothed_position[1]) > mouse_update_threshold:
                    pyautogui.moveTo(smoothed_position[0], smoothed_position[1])
                    last_mouse_position = smoothed_position

                # Detect clicks
                distance_thumb_index = calculate_distance(index_finger_tip, thumb_tip)
                if distance_thumb_index < 0.05:
                    pyautogui.click()

            # Draw hand landmarks on the frame
            # Uncomment the following line to show the video with hand skeletons
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame, last_mouse_position, smoothed_position, last_scroll_position, smoothed_scroll_position

def main():
    last_mouse_position = (0, 0)
    smoothed_position = (0, 0)
    last_scroll_position = 0
    smoothed_scroll_position = 0
    prev_time = time.time()
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame directly
        frame, last_mouse_position, smoothed_position, last_scroll_position, smoothed_scroll_position = process_frame(
            frame, last_mouse_position, smoothed_position, last_scroll_position, smoothed_scroll_position)

        # Limit the frame rate
        elapsed_time = time.time() - prev_time
        if elapsed_time < frame_delay:
            time.sleep(frame_delay - elapsed_time)
        prev_time = time.time()

        # Increment frame count
        frame_count += 1

        # Calculate and print FPS every second
        if time.time() - start_time >= 1.0:
            fps = frame_count / (time.time() - start_time)
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

        # Show the frame
        # Uncomment the following line to display the video
        # cv2.imshow('Hand Tracking', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()