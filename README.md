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
