import cv2
import mediapipe as mp
import numpy as np

# This line gets the specific 'pose' model from MediaPipe
mp_pose = mp.solutions.pose

# This line turns on the pose model so we can use it
pose = mp_pose.Pose()

# This will be used later to draw the skeleton on the screen
mp_drawing = mp.solutions.drawing_utils