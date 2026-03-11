import numpy as np

# 3D World Coordinates (in meters)
# Origin (0,0,0) is at the center of the pitch.
# Length (X): 20.12m (Bowling Crease to Bowling Crease)
# Width (Y): 3.05m (Return Crease to Return Crease)
# Note: homography_pipeline.py defined origin at Batting Left Corner. 
# BroadTrack implementation usually prefers center.
# Let's stick to Center Origin to be consistent with visualization scripts.

PITCH_LENGTH = 20.12
PITCH_WIDTH = 3.05

# Keypoints from YOLO model (8 points)
# X axis: Length (Long), -X = Batting end, +X = Bowling end
# Y axis: Width (Short), -Y = Left, +Y = Right
# Half-length L = 10.06m, Half-width W = 1.525m
# Popping crease is 1.22m in front of each bowling crease (POP = 1.22m)
#
# Points 0-3: bowling crease × return crease intersections (pitch corners)
# Points 4-7: popping crease × return crease intersections

POP_CREASE_OFFSET = 1.22

OBJECT_POINTS = np.array([
    [-10.06, -1.525, 0],  # 0: batting_left_corner
    [-10.06,  1.525, 0],  # 1: batting_right_corner
    [ 10.06,  1.525, 0],  # 2: bowling_right_corner
    [ 10.06, -1.525, 0],  # 3: bowling_left_corner
    [ -8.84, -1.525, 0],  # 4: batting_left_pop   (-10.06 + 1.22)
    [ -8.84,  1.525, 0],  # 5: batting_right_pop  (-10.06 + 1.22)
    [  8.84,  1.525, 0],  # 6: bowling_right_pop  (+10.06 - 1.22)
    [  8.84, -1.525, 0],  # 7: bowling_left_pop   (+10.06 - 1.22)
], dtype=np.float32)

NUM_KEYPOINTS = 8

# Keypoint Indices in YOLO output
KP_BAT_LEFT = 0
KP_BAT_RIGHT = 1
KP_BOW_RIGHT = 2
KP_BOW_LEFT = 3
KP_BAT_LEFT_POP = 4
KP_BAT_RIGHT_POP = 5
KP_BOW_RIGHT_POP = 6
KP_BOW_LEFT_POP = 7

# Visualization Colors (BGR)
COLOR_PITCH = (0, 255, 0)
COLOR_CREASE = (255, 0, 0)
COLOR_STUMP = (0, 0, 255)
