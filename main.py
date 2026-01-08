import cv2
import mediapipe as mp
import pyautogui
import math

# -------- SYSTEM SAFETY FIX --------
pyautogui.FAILSAFE = False
PINCH_DISTANCE = 80
RELEASE_DISTANCE = 130
# ----------------------------------

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
drawing_utils = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

index_x, index_y = 0, 0
clicked = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.process(rgb)
    hands = result.multi_hand_landmarks

    if hands:
        for hand_landmarks in hands:
            drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark

            index_found = False
            thumb_pos = None

            # ---------- LOOP 1: READ LANDMARKS ----------
            for id, lm in enumerate(landmarks):
                x = int(lm.x * w)
                y = int(lm.y * h)

                if x < 0 or y < 0 or x > w or y > h:
                    continue

                # INDEX FINGER
                if id == 8:
                    index_found = True
                    cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)

                    index_x = screen_width / w * x
                    index_y = screen_height / h * y
                    pyautogui.moveTo(index_x, index_y, duration=0.01)

                # THUMB (ALWAYS DRAW)
                if id == 4:
                    cv2.circle(frame, (x, y), 15, (255, 0, 0), -1)
                    thumb_pos = (
                        screen_width / w * x,
                        screen_height / h * y
                    )

            # ---------- LOOP 2: CLICK LOGIC ----------
            if index_found and thumb_pos:
                thumb_x, thumb_y = thumb_pos
                distance = math.hypot(index_x - thumb_x,
                                       index_y - thumb_y)

                # PINCH → CLICK (one time)
                if distance < PINCH_DISTANCE and not clicked:
                    pyautogui.mouseDown()
                    pyautogui.mouseUp()
                    clicked = True
                    print("CLICK")

                # RELEASE → RESET
                if distance > RELEASE_DISTANCE:
                    clicked = False
    else:
        clicked = False

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
