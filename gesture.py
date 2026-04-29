import cv2
import mediapipe as mp
import math
import time
import requests     
import threading
from pynput.keyboard import Key, Controller
keyboard = Controller()
last_gesture_time = 0
COOLDOWN = 1.5

last_stats_time = 0
STATS_COOLDOWN = 2.0
anki_due = None

current_gesture = "NONE"
gesture_hold_start = 0
HOLD_DURATION = 1  # seconds gesture must be held before firing

def fetch_anki_stats():
    global anki_due
    anki_due = get_anki_stats()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

import urllib.request
import os
if not os.path.exists("hand_landmarker.task"):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "hand_landmarker.task"
    )
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Hand Tracking")

finger_ratio = 1.3
thumb_thresh = 0.03

FINGERS = [
    [8,  5],   # index:  [tip, MCP]
    [12, 9],   # middle: [tip, MCP]
    [16, 13],  # ring:   [tip, MCP]
    [20, 17],  # pinky:  [tip, MCP]
]

def dist2d(a, b): # pythagoras theorem, (a and b are landmarks so a.x is the x coordinate)
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def is_extended(hand, tip_idx, mcp_idx, ratio_thresh):
    wrist = hand[0]
    tip_d = dist2d(hand[tip_idx], wrist) #distance to tip
    mcp_d = dist2d(hand[mcp_idx], wrist) # distance to MCP knuckle
    ratio_ok = (tip_d / mcp_d if mcp_d > 0 else 1.0) > ratio_thresh  # the ratio between tip_d and mcp_d is 1.6 when extended or 0.9 when closed
    depth_ok = hand[tip_idx].z < hand[mcp_idx].z - 0.05 #if the finger tip is substantially closer to the camera than the mcp, count it as extended
    return ratio_ok or depth_ok

# Calibration state
calib_phase = None   # None | 'fist' | 'open'
CALIB_FRAMES = 45
calib_samples = {'fist': [], 'open': []}


def collect_sample(hand):
    """Returns (finger ratios, thumb x-spread)."""
    wrist = hand[0] # reference point
    ratios = []
    for tip_idx, mcp_idx in FINGERS: #each loop focuses on one finger
        tip_d = dist2d(hand[tip_idx], wrist)
        mcp_d = dist2d(hand[mcp_idx], wrist)
        ratios.append(tip_d / mcp_d if mcp_d > 0 else 1.0) # same calculations as 'is_extended'
    thumb_spread = abs(hand[4].x - hand[2].x) # horizontal distance between thumb joints
    return ratios, thumb_spread # returns a list of 4 numbers + a thumb number 


def apply_calibration():
    global finger_ratio, thumb_thresh
    fist = calib_samples['fist']
    open_ = calib_samples['open']

    fist_ratio = sum(sum(s[0]) / 4 for s in fist) / len(fist) # average finger ratio when hand is a fist
    open_ratio = sum(sum(s[0]) / 4 for s in open_) / len(open_) # average finger ratio when hand is open
    finger_ratio = (fist_ratio + open_ratio) / 2 # average of the 2, this is the threshold: anything bigger --> open , smaller --> closed

    fist_thumb = sum(s[1] for s in fist) / len(fist) 
    open_thumb = sum(s[1] for s in open_) / len(open_)
    thumb_thresh = (fist_thumb + open_thumb) / 2 # these thumb calculations work the same
 
def handle_gesture(gesture):
    global last_gesture_time, current_gesture, gesture_hold_start

    if gesture == "NONE":
        current_gesture = "NONE"
        gesture_hold_start = 0
        return

    # If gesture changed, reset the hold timer
    if gesture != current_gesture:
        current_gesture = gesture
        gesture_hold_start = time.time()
        return

    # Same gesture — check if held long enough
    held_for = time.time() - gesture_hold_start
    if held_for < HOLD_DURATION:
        return

    # Check cooldown
    if time.time() - last_gesture_time < COOLDOWN:
        return

    last_gesture_time = time.time()
    gesture_hold_start = time.time()  # reset hold so it doesn't keep firing

    if gesture == "SHOW_ANSWER":
        keyboard.press(Key.space)
        keyboard.release(Key.space)
    elif gesture == "AGAIN":
        keyboard.press('1')
        keyboard.release('1')
    elif gesture == "HARD":
        keyboard.press('3')
        keyboard.release('3')
    elif gesture == "EASY":
        keyboard.press('4')
        keyboard.release('4')

def get_gesture(hand, finger_ratio=1.3, thumb_thresh=0.03):
    extended = [is_extended(hand, tip, mcp, finger_ratio) for tip, mcp in FINGERS] # get a list of like [True, True, False, True] by doing is_extended for each finger
    thumb_extended = abs(hand[4].x - hand[2].x) > thumb_thresh # if the thumb gap is large enough returns true

    index, middle, ring, pinky = extended # turns the list into 4 variables

    if index and middle and ring and pinky and thumb_extended: #defines all the gestures
        return "SHOW_ANSWER"
    elif thumb_extended and not any(extended):
        return "AGAIN"
    elif index and middle and not ring and not pinky and not thumb_extended:
        return "EASY"
    elif index and not middle and not ring and not pinky and not thumb_extended:
        return "HARD"
    else:
        return "NONE"

def get_anki_stats():
    try:
        card_response = requests.post('http://localhost:8765', json={
            "action": "guiCurrentCard",
            "version": 6
        })
        card_data = card_response.json()
        if card_data['result'] is None:
            return None
        deck_name = card_data['result']['deckName']

        stats_response = requests.post('http://localhost:8765', json={
            "action": "getDeckStats",
            "version": 6,
            "params": {"decks": [deck_name]}
        })
        stats = list(stats_response.json()['result'].values())[0]
        return {
            'due': stats['review_count'],
            'learn': stats['learn_count'],
            'new': stats['new_count']
        }
    except:
        return None

def draw_hold_bar(frame, gesture, hold_start):
    if gesture == "NONE" or hold_start == 0:
        return
    
    progress = min((time.time() - hold_start) / HOLD_DURATION, 1.0)
    
    bar_x, bar_y = 10, frame.shape[0] - 50
    bar_width = 300
    bar_height = 12
    
    # Background bar (dark)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (50, 50, 50), -1)
    
    # Filled portion - colour changes from blue to green as it fills
    fill_width = int(bar_width * progress)
    r = int(0 + (0 * progress))
    g = int(150 + (105 * progress))
    b = int(255 - (255 * progress))
    if fill_width > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                      (b, g, r), -1)
    
    # White outline
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (255, 255, 255), 1)

with HandLandmarker.create_from_options(options) as landmarker: #main part-----------------------------------
    while True:
        ret, frame = cap.read() # capture a frame
        frame = cv2.flip(frame, 1) # mirror frame so it acts as mirror
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert BGR to RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image) #result = landmark positions for that frame

        if result.hand_landmarks: # if a hand was found
            for hand in result.hand_landmarks: 
                coords = []
                for landmark in hand: # for each landmark
                    h, w, _ = frame.shape # height, width, colour channels (discarded)
                    cx, cy = int(landmark.x * w), int(landmark.y * h) 
                    coords.append((cx, cy)) # coords[8] is the pixel position of landmark 8
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1) #draws a filled circle that the coordinates cx,cy (which frame to draw on, centre position, radius in pixels, Color in BGR, -1 = filled)

                if calib_phase in ('fist', 'open'): # tracks what phase the calibration is in
                    calib_samples[calib_phase].append(collect_sample(hand)) #adds the sample to whichever list is currently active

                gesture = get_gesture(hand, finger_ratio, thumb_thresh) # figures out what the gesture is
                handle_gesture(gesture) # HANDLES it
                cv2.putText(frame, gesture, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3) # arguments: (image to draw on, string to display, pixel position, font, font scale, colour BGR, thickness)

                connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS #Hand_connections contains a list of landmarks that should be connected
                for connection in connections: #for each pair, join them together
                    start = coords[connection.start]
                    end = coords[connection.end]
                    cv2.line(frame, start, end, (255, 255, 255), 2) #draws a line 

        if calib_phase == 'fist': #during the fist phase
            n = len(calib_samples['fist']) # how many samples have been collected while first
            cv2.putText(frame, f"CALIBRATE: Hold CLOSED FIST ({CALIB_FRAMES - n})",
                        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2) # text styling
            if n >= CALIB_FRAMES:
                calib_phase = 'open' # swap to open hand calibration phase
                calib_samples['open'] = []
        elif calib_phase == 'open':
            n = len(calib_samples['open']) #again how many samples have been collected while open
            cv2.putText(frame, f"CALIBRATE: Hold OPEN HAND ({CALIB_FRAMES - n})",
                        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2) #text stylinh
            if n >= CALIB_FRAMES: 
                apply_calibration() # self explanatory
                calib_phase = None
        else:
            cv2.putText(frame, "Press C to calibrate | Q to quit",
                        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1) # text styling
        if time.time() - last_stats_time > STATS_COOLDOWN:
            threading.Thread(target=fetch_anki_stats, daemon=True).start()
            last_stats_time = time.time()

        if anki_due is not None:
            cv2.putText(frame, f"Due: {anki_due['due']}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Learn: {anki_due['learn']}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"New: {anki_due['new']}", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 200), 2)
        draw_hold_bar(frame, current_gesture, gesture_hold_start)
        cv2.imshow("Hand Tracking", frame)
        key = cv2.waitKey(1) & 0xFF # wait 1ms for a keypress
        if key == ord('q'): #converts q into ascii and compares it to any key pressed
            break 
        elif key == ord('c') and calib_phase is None: #if not currently calibrating and c is pressed
            calib_phase = 'fist' # start the calibration by setting it to fist
            calib_samples = {'fist': [], 'open': []} # clears previous calibration samples

cap.release() # if main loop breaks, stop using the webcam
cv2.destroyAllWindows() # and close display window
