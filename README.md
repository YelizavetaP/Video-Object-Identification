# Video-Object-Identification

Smile detection on streaming video with Haar cascades.

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
   to distinguish it from the green frontal box. The profile cascade is trained
   for one side only (right-facing faces), so it is run twice — once on the
   frame and once on a horizontally flipped copy — and the flipped detections
   are mapped back (`frame_width - px - pw`) to also catch the other side.

6. **Suppress profile boxes that overlap a frontal face** — the profile cascade
   also fired when the man faced the camera, drawing a blue box on top of the
   green frontal one. Frontal detection is more reliable, so it wins: an
   `overlaps()` helper (intersection over the smaller box) drops any profile box
   that overlaps a detected frontal face, while genuine profile views — where no
   frontal face is detected — are still drawn.

## Additionally implemented

- **Live camera mode (`camera.py`)** — a standalone script that runs the same
  detection pipeline (green frontal, blue profile, red smile, with the overlap
  suppression) on the laptop webcam instead of a video file. It opens the camera
  with the DirectShow backend (`cv2.VideoCapture(0, cv2.CAP_DSHOW)`) for fast
  startup on Windows, checks that the camera opened, and skips dropped frames.
  Run it with `python camera.py` and press `Esc` to quit.
