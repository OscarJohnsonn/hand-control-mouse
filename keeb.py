import cv2
import mediapipe as mp
from pynput.keyboard import Controller

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

keyboard = Controller()

key_labels = [
    'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P',
    'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L',
    'Z', 'X', 'C', 'V', 'B', 'N', 'M'
]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    height, width, _ = image.shape

    keys = []
    key_width = width // 10
    key_height = height // 3
    for i, label in enumerate(key_labels):
        if i < 10:
            x = i * key_width + key_width // 2
            y = key_height // 2
        elif i < 19:
            x = (i - 10) * key_width + key_width // 2 + key_width // 2
            y = key_height + key_height // 2
        else:
            x = (i - 19) * key_width + key_width // 2 + key_width
            y = 2 * key_height + key_height // 2
        keys.append({'key': label, 'pos': (x, y)})

    for key in keys:
        cv2.putText(image, key['key'], key['pos'], cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(image, key['pos'], 50, (255, 0, 0), 3)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            pointer_finger_tip = hand_landmarks.landmark[8]
            lm_x, lm_y = int(pointer_finger_tip.x * width), int(pointer_finger_tip.y * height)
            for key in keys:
                key_x, key_y = key['pos']
                if (key_x - 50 < lm_x < key_x + 50) and (key_y - 50 < lm_y < key_y + 50):
                    cv2.putText(image, f"Pressed: {key['key']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                    keyboard.press(key['key'])
                    keyboard.release(key['key'])

    cv2.imshow('Virtual Keyboard', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()