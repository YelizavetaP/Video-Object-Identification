# ----------- Pattern recognition: object detection and tracking -----------------

'''

Face / profile / smile detection on the live laptop camera with Haar cascades.

Same detection logic as main.py, but the video source is the webcam instead of
a file. Press 'Esc' to quit.

'''

import os
import cv2

_here = os.path.dirname(os.path.abspath(__file__))

# Load the Haar cascade files for the face, the profile and the smile.
# The smile cascade is trained to find a mouth inside an already-cropped face,
# so it must run on a face ROI -- never on the whole frame, or it fires on
# eyes, nostrils and other mouth-like patterns and draws false boxes.
face_cascade = cv2.CascadeClassifier(
        os.path.join(_here, 'haarcascade_frontalface_default.xml'))
profile_cascade = cv2.CascadeClassifier(
        os.path.join(_here, 'haarcascade_profileface.xml'))
smile_cascade = cv2.CascadeClassifier(
        os.path.join(_here, 'haarcascade_smile.xml'))


# Check if the cascade files have been loaded correctly
if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')
if profile_cascade.empty():
	raise IOError('Unable to load the profile face cascade classifier xml file')
if smile_cascade.empty():
	raise IOError('Unable to load the smile cascade classifier xml file')


def overlaps(a, b, threshold=0.3):
    """Return True if rectangles a and b (x, y, w, h) overlap enough to be
    treated as the same face. Uses intersection over the smaller box, which is
    robust when the two detections differ in size."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_w * inter_h
    if inter == 0:
        return False
    return inter / min(aw * ah, bw * bh) > threshold


# Initialize the video capture object from the laptop camera.
# CAP_DSHOW is the DirectShow backend, which opens the webcam faster on Windows.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Make sure the camera actually opened
if not cap.isOpened():
    raise IOError('Unable to open the laptop camera')

# Define the scaling factor
scaling_factor = 1.0

# Iterate until the user hits the 'Esc' key
while True:
    # Capture the current frame
    ret, frame = cap.read()

    # A camera can drop frames; skip the iteration if no frame was returned
    if not ret:
        continue

    # Resize the frame
    if scaling_factor != 1.0:
        frame = cv2.resize(frame, None,
                fx=scaling_factor, fy=scaling_factor,
                interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''
    Ідентифікація облич на потоковому відео з каскадами Хаара
    '''

    # Run the face detector on the grayscale image
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each detected face, look for a smile only inside that face
    for (x,y,w,h) in face_rects:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

        # Restrict the smile search to the lower half of the face ROI,
        # where the mouth is -- this avoids matches on eyes and forehead.
        roi_gray = gray[y + h // 2:y + h, x:x + w]
        roi_color = frame[y + h // 2:y + h, x:x + w]

        # High minNeighbors (and a larger scaleFactor) keeps only strong,
        # repeated detections, filtering out the noisy false positives.
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)

        # A face has only one mouth, but the detector can return several
        # overlapping boxes for it. Keep just the largest one and draw it.
        if len(smiles) > 0:
            sx, sy, sw, sh = max(smiles, key=lambda r: r[2] * r[3])
            cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (0,0,255), 2)

    '''
    Ідентифікація обличчя в профіль на потоковому відео з каскадами Хаара
    '''

    # The profile cascade is trained for one side only (right-facing faces),
    # so it would miss a face turned the other way. Run it once on the frame
    # and once on a horizontally flipped copy, then map the flipped detections
    # back to the original coordinates.
    profiles = list(profile_cascade.detectMultiScale(gray, 1.3, 5))
    frame_width = gray.shape[1]
    for (px,py,pw,ph) in profile_cascade.detectMultiScale(cv2.flip(gray, 1), 1.3, 5):
        profiles.append((frame_width - px - pw, py, pw, ph))

    # Draw a rectangle around each profile face in a different colour (blue).
    # Frontal detection is more reliable, so when the face is head-on and the
    # profile cascade fires on the same face, skip it -- the frontal box
    # already covers it.
    for (px,py,pw,ph) in profiles:
        if any(overlaps((px,py,pw,ph), f) for f in face_rects):
            continue
        cv2.rectangle(frame, (px,py), (px+pw,py+ph), (255,0,0), 3)

    # Display the output
    cv2.imshow('Face Detector (camera)', frame)

    # Check if the user hit the 'Esc' key
    c = cv2.waitKey(1)
    if c == 27:
        break

# Release the video capture object
cap.release()

# Close all the windows
cv2.destroyAllWindows()
