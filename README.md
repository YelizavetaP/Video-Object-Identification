# Video-Object-Identification

Smile detection on streaming video with Haar cascades.

## Task

- **Group 3** — a personally selected object.
- **Level 1** — only one group of objects.

The object I selected is a **smile on a face**. I additionally added **profile
face** detection to improve the results.

## Project structure

```
Video-Object-Identification/
├── main.py                                  # detection on a video file (media/smile2.mp4)
├── camera.py                                # same detection on the live laptop camera
├── haarcascade_frontalface_default.xml      # frontal-face cascade (green box)
├── haarcascade_profileface.xml              # profile-face cascade (blue box)
├── haarcascade_smile.xml                    # smile cascade (red box)
├── media/                                   # input videos
│   ├── smile.mp4
│   ├── smile2.mp4
│   └── 01.mp4 … 04.mp4
├── task5.pdf                                # task description
└── README.md
```

- **`main.py`** — reads a video from `media/`, runs the full detection pipeline
  and shows the result. This is the main deliverable.
- **`camera.py`** — the same pipeline on the live webcam (see *Additionally
  implemented* below).
- **`haarcascade_*.xml`** — the pre-trained Haar cascade classifiers used for
  detection.
- **`media/`** — the input videos the detector runs on.

## Approach and iterations

Getting reliable smile detection took three attempts, each fixing the false
positives left by the previous one:

1. **Smile cascade on the whole frame** — far too many false positives. The
   smile cascade is trained on a cropped mouth, so scanning a full image makes
   it fire on eyes, nostrils, shadows and other mouth-like patterns.

2. **Nested detection (face first, then smile inside the face ROI)** — fewer
   false positives, but still a lot *within* each face. Searching the entire
   face region lets the cascade match on eyes, eyebrows and the forehead, and
   the default `detectMultiScale` parameters (`scaleFactor=1.1, minNeighbors=3`)
   are too permissive to filter the noise.

3. **Smile search restricted to the lower half of the face ROI, with stricter
   parameters** (`scaleFactor=1.7, minNeighbors=22`) — this is the working
   version. Cropping to the lower half removes the eye/forehead matches
   structurally, and the high `minNeighbors` keeps only strongly confirmed
   detections.

4. **Keep only the largest smile box per face** — after step 3 the smile was
   detected correctly, but the detector still returned several overlapping
   rectangles for the same mouth (a smaller box nested inside the correct one).
   Since a face has only one mouth, the code now picks the single largest
   detection by area (`max(smiles, key=lambda r: r[2] * r[3])`) and draws just
   that one.

5. **Add profile-face detection (blue box)** — in the video the man is first
   shown from the side, but the frontal cascade only detects faces head-on, so
   the side view was missed. Added `haarcascade_profileface.xml`, drawn in blue
   to distinguish it from the green frontal box. 

6. **Suppress profile boxes that overlap a frontal face** — the profile cascade
   also fired when the man faced the camera, drawing a blue box on top of the
   green frontal one. Frontal detection is more reliable for next smile detection, so it wins: an
   `overlaps()` helper (intersection over the smaller box) drops any profile box
   that overlaps a detected frontal face.

## Additionally implemented

- **Live camera mode (`camera.py`)** — a standalone script that runs the same
  detection pipeline (green frontal, blue profile, red smile, with the overlap
  suppression) on the laptop webcam instead of a video file.
