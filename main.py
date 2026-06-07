# ----------- Pattern recognition: object detection and tracking -----------------

'''

Ідентифікація облич на потоковому відео

алгоритм ідентифікації "каскади Хаара":
 - класифікатор Хаара - ієрархічна множина простих вирішувачів - як сума та різниця підобластей зображення;
 - надмірність даних - потокове відео, класифікатор Хаара - вважається вже сформованим;
 - каскади Хаара складаються з множини "цифрових автоматів" що побудовані в ієрархію структур "дерево";
 - класифікатор на кінцевому прошарку  має образ (образи) із часткових ознак;
 - зображення описується множиною частковимих ознак;
 - процес ідентифікації: порівняння множини ознак на прошарках для віднесення обєкту до найближчого образа.

Prateek Joshi Artificial Intelligence applications with Python, Part 13:
https://mazz.keybase.pub/ebooks/ai/9781786464392-ARTIFICIAL_INTELLIGENCE_WITH_PYTHON.pdf
Scripts:
https://github.com/PacktPublishing/Artificial-Intelligence-with-Python

Package            Version
------------------ -----------
opencv-python      3.4.18.65


'''

import os
import cv2

_here = os.path.dirname(os.path.abspath(__file__))
_media = os.path.join(_here, 'media')

# Load the Haar cascade files for the face and the smile.
# The smile cascade is trained to find a mouth inside an already-cropped face,
# so it must run on a face ROI -- never on the whole frame, or it fires on
# eyes, nostrils and other mouth-like patterns and draws false boxes.
face_cascade = cv2.CascadeClassifier(
        os.path.join(_here, 'haarcascade_frontalface_default.xml'))
smile_cascade = cv2.CascadeClassifier(
        os.path.join(_here, 'haarcascade_smile.xml'))


# Check if the cascade files have been loaded correctly
if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')
if smile_cascade.empty():
	raise IOError('Unable to load the smile cascade classifier xml file')

# Initialize the video capture object from the webcam
# cap = cv2.VideoCapture(0)

# Initialize the video capture object
cap = cv2.VideoCapture(os.path.join(_media, 'smile2.mp4'))
# cap = cv2.VideoCapture(0)

# Define the scaling factor
scaling_factor = 0.5

# Iterate until the user hits the 'Esc' key
while True:
    # Capture the current frame
    _, frame = cap.read()

    # Resize the frame
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

        # Draw a rectangle around the smile (relative to the face ROI)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (0,0,255), 2)

    # Display the output
    cv2.imshow('Face Detector', frame)

    # Check if the user hit the 'Esc' key
    c = cv2.waitKey(1)
    if c == 27:
        break

# Release the video capture object
cap.release()

# Close all the windows
cv2.destroyAllWindows()
