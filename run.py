import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter
import math

# ------------------------------
# Load model (English only)
# ------------------------------
model_en = pickle.load(open('modelKADER.p', 'rb'))['modelKADER']

# ------------------------------
# Labels (4 directions)
# ------------------------------
labels_dict_en = {5: 'F', 1: 'B', 11: 'L', 17: 'R'}

# ------------------------------
# Camera
# ------------------------------
cap = cv2.VideoCapture(0)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# ------------------------------
# Prediction tracking
# ------------------------------
predictions_queue = deque(maxlen=20)
most_common_char = ""
last_active = None
pulse_start_time = 0

# ------------------------------
# Draw Arrow Helper
# ------------------------------
def draw_arrow(frame, label, center, color, active=False, pulse_scale=1.0):
    x, y = center
    thickness = int(3 * pulse_scale) if active else 2
    arrow_size = int(18 * pulse_scale)
    arrow_color = color

    if label == 'F':  # ↑
        pts = np.array([[x, y - arrow_size], [x - 12, y + 10], [x + 12, y + 10]], np.int32)
    elif label == 'B':  # ↓
        pts = np.array([[x, y + arrow_size], [x - 12, y - 10], [x + 12, y - 10]], np.int32)
    elif label == 'L':  # ←
        pts = np.array([[x - arrow_size, y], [x + 10, y - 12], [x + 10, y + 12]], np.int32)
    elif label == 'R':  # →
        pts = np.array([[x + arrow_size, y], [x - 10, y - 12], [x - 10, y + 12]], np.int32)

    # خلفية دائرية صغيرة
    cv2.circle(frame, (x, y), int(30 * pulse_scale), (255, 255, 255), -1)
    cv2.circle(frame, (x, y), int(30 * pulse_scale), arrow_color, thickness)
    cv2.fillPoly(frame, [pts], arrow_color)

    # لون الحرف (أحمر عند التفعيل، رمادي عند الخمول)
    text_color = (0, 0, 255) if active else (160, 160, 160)

    cv2.putText(frame, label, (x - 10, y + int(45 * pulse_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * pulse_scale, text_color, 2)
    return frame


# ------------------------------
# Main loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # ---------------- Prediction ----------------
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        x_, y_ = [], []
        data_aux = []

        for landmark in hand_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)

        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))

        prediction = model_en.predict([np.asarray(data_aux)])
        predicted_character = labels_dict_en[int(prediction[0])]

        predictions_queue.append(predicted_character)
        most_common_char, count = Counter(predictions_queue).most_common(1)[0]

        if most_common_char != last_active:
            pulse_start_time = time.time()
            last_active = most_common_char
    else:
        most_common_char = ""

    # ---------------- Draw arrows ----------------
    base_x = W - 200
    base_y = H - 150
    spacing = 80

    colors = {'F': (160, 160, 160), 'B': (160, 160, 160),
              'L': (160, 160, 160), 'R': (160, 160, 160)}

    if most_common_char in colors:
        colors[most_common_char] = (0, 255, 0)

    pulse_scale = 1.0
    if most_common_char:
        elapsed = time.time() - pulse_start_time
        pulse_scale = 1 + 0.1 * math.sin(elapsed * 6)

    frame = draw_arrow(frame, 'F', (base_x + spacing, base_y - spacing),
                       colors['F'], most_common_char == 'F',
                       pulse_scale if most_common_char == 'F' else 1.0)
    frame = draw_arrow(frame, 'B', (base_x + spacing, base_y + spacing),
                       colors['B'], most_common_char == 'B',
                       pulse_scale if most_common_char == 'B' else 1.0)
    frame = draw_arrow(frame, 'L', (base_x, base_y),
                       colors['L'], most_common_char == 'L',
                       pulse_scale if most_common_char == 'L' else 1.0)
    frame = draw_arrow(frame, 'R', (base_x + 2 * spacing, base_y),
                       colors['R'], most_common_char == 'R',
                       pulse_scale if most_common_char == 'R' else 1.0)

    cv2.imshow("Direction Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
