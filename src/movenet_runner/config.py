# Inference/input config
WIDTH = HEIGHT = 256
SCORE_THRESH = 0.11           # per-keypoint min confidence to draw
MAX_PERSONS = 6               # MoveNet MultiPose Lightning returns up to 6 persons
NUM_KEYPOINTS = 17            # COCO-17

# Colors (BGR)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
KEYPOINT_COLOR = (255, 0, 0)

# Skeleton edges per COCO-17 indexing used by MoveNet (y, x, score per kp)
# 0: nose, 1: left eye, 2: right eye, 3: left ear, 4: right ear,
# 5: left shoulder, 6: right shoulder, 7: left elbow, 8: right elbow,
# 9: left wrist, 10: right wrist, 11: left hip, 12: right hip,
# 13: left knee, 14: right knee, 15: left ankle, 16: right ankle
EDGE_COLORS = {
    (0, 1): MAGENTA, (0, 2): CYAN, (1, 3): MAGENTA, (2, 4): CYAN,
    (0, 5): MAGENTA, (0, 6): CYAN, (5, 7): MAGENTA, (7, 9): CYAN,
    (6, 8): MAGENTA, (8, 10): CYAN, (5, 6): MAGENTA, (5, 11): CYAN,
    (6, 12): MAGENTA, (11, 12): CYAN, (11, 13): MAGENTA, (13, 15): CYAN,
    (12, 14): MAGENTA, (14, 16): CYAN
}
