import cv2
import mediapipe as mp
import numpy as np
import time
import math

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

exercises = ["bicep_curl", "shoulder_press", "squat"]
current_exercise_index = 0
exercise = exercises[current_exercise_index]

counter = 0
stage = None
rep_limit = 12

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark
        
        if exercise == "bicep_curl":
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle = calculate_angle(shoulder, elbow, wrist)

            # Rep counting logic with debounce
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == "down":
                stage = "up"
                counter += 1
                print(f"Bicep Curl Reps: {counter}")

        elif exercise == "shoulder_press":
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle = calculate_angle(shoulder, elbow, wrist)

            if angle < 30:
                stage = "down"
            if angle > 100 and stage == "down":
                stage = "up"
                counter += 1
                print(f"Shoulder Press Reps: {counter}")

        elif exercise == "squat":
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            angle = calculate_angle(hip, knee, ankle)

            if angle > 160:
                stage = "down"
            if angle < 90 and stage == "down":
                stage = "up"
                counter += 1
                print(f"Squat Reps: {counter}")

        # Position to show angle
        if exercise == "bicep_curl":
            pos = tuple(np.multiply(elbow, [640, 480]).astype(int))
        elif exercise == "shoulder_press":
            pos = tuple(np.multiply(elbow, [640, 480]).astype(int))
        elif exercise == "squat":
            pos = tuple(np.multiply(knee, [640, 480]).astype(int))
        else:
            pos = (50, 50)

        cv2.putText(image, str(int(angle)), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    
    except:
        pass

    # Switch exercise after rep_limit reached
    if counter >= rep_limit:
        counter = 0
        stage = None
        current_exercise_index += 1
        if current_exercise_index >= len(exercises):
            current_exercise_index = 0
        exercise = exercises[current_exercise_index]

    # Animated rep count
    t = time.time()
    bounce_scale = 2 + 0.5 * math.sin(t * 6)
    x, y = 50, 100
    shadow_offset = 3

    cv2.rectangle(image, (0, 0), (280, 120), (245, 117, 16), -1)

    # Shadow
    cv2.putText(image, str(counter), (x + shadow_offset, y + shadow_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, bounce_scale, (0, 0, 0), int(bounce_scale * 3), cv2.LINE_AA)
    # Foreground count
    cv2.putText(image, str(counter), (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, bounce_scale, (0, 255, 255), int(bounce_scale * 3), cv2.LINE_AA)

    # Exercise name and stage
    cv2.putText(image, "Exercise:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(image, exercise.replace('_', ' ').title(), (130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(image, "Stage:", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(image, stage if stage else "None", (130, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('AI Coach', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
