# Document Scanner (RPi 5 + Picamera2 + OpenCV) — README

<img width="1532" height="887" alt="image" src="https://github.com/user-attachments/assets/fa0839bd-9b3b-4a50-9a70-a739bbc2128c" />

Robust, orientation-agnostic document detection and scan warping on Raspberry Pi 5 (IMX500 + Picamera2), with optional OCR and auto-rotation.

## What this project does (at a glance)

 - Captures RGB888 frames from Picamera2 and pre-processes them for stable edges across lighting/orientation. 
 - Builds an edge map (Canny) and stabilizes borders via morphological closing + dilation. 
 - Extracts external contours only, shortlists by rotation-invariant size & aspect using minAreaRect, and rejects text/grout via extent and solidity. 
 - Forces a true 4-corner page polygon by approximating the convex hull (approxPolyDP, Douglas–Peucker). 
 - Computes a 4-point perspective transform and warps to a top-down scan. 
 - (Optional) Uses Tesseract OSD for orientation correction (0/90/180/270) with short history (debounce). 

## Techniques, why they were chosen, and alternatives
1. Local contrast normalization (CLAHE) → Gaussian blur
CLAHE boosts text/background separation under uneven indoor lighting; blur reduces high-frequency noise before Canny. 

Alternatives:
Global histogram equalization (simpler, but can wash out highlights) 
Color-space thresholding (e.g., LAB ‘L’ channel) — workable but more scene-specific tuning.
Reason not used instead: CLAHE is more robust across floor colors/tiles and shadows than a global equalize or fixed thresholds. 

2. Canny edges + morphology (closing, dilation)
Canny’s hysteresis gives clean, thin borders; closing seals gaps at steep rotations; dilation reconnects broken border segments. 
Alternatives:
Sobel/Scharr gradient magnitude + threshold (more tuning),
HED/learned edge detectors (heavy for RPi).
Reason not used instead: Canny + simple morphology hits the speed/quality sweet spot on Raspberry Pi. 

3. Contour retrieval (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
Only the outermost shapes matter for a page; skipping child contours avoids selecting text inside the page. CHAIN_APPROX_SIMPLE reduces memory/CPU. 
Alternatives: RETR_TREE with hierarchy logic.
Reason not used: Full hierarchies are slower and unnecessary once we only want the page boundary. 

4. Rotation-invariant gating via minAreaRect, then extent & solidity
minAreaRect gives width/height regardless of rotation; extent = area / (bbox area) filters long skinny grout lines; solidity = area / (hull area) rejects text clusters / non-solid shapes. 
Alternatives:
Pure area threshold (fails on angled pages),
Hough lines + rectangle assembly (fragile with textured floors).
Reason not used: Min-area rectangle features are simpler and robust to rotation; extent/solidity come straight from classic contour features. 

5. Convex-hull → adaptive approxPolyDP to force 4 corners
Pages are convex; approximating the hull avoids interior wiggles from text/texture. approxPolyDP (Douglas–Peucker) yields a clean quad when epsilon is adapted moderately. 
Alternatives:
boxPoints(minAreaRect) (fast but not the true page if one edge is curved/occluded),
Line-segment detectors (LSD/EDLines) + corner finding (heavier; parameter-sensitive).
Reason not used: Hull+DP is fast, stable, and tracks the real boundary better than a pure oriented box. 

6. 4-point perspective warp (getPerspectiveTransform → warpPerspective)
Exact homography from ordered TL-TR-BR-BL points produces a metric-correct “scan”. 
Alternatives: Affine warp (insufficient for perspective foreshortening).
Reason not used: A4/sheet-like documents require a full projective transform. 

7. Orientation with Tesseract OSD (debounced majority)
OSD returns discrete rotations (0/90/180/270) + confidence; a short history suppresses flip-flop near ambiguous angles. 
Alternatives: Hough-based baseline voting / gradient-energy ratios; good backups but OSD is language-aware and simple to integrate. 

🔁 Processing sequence (exact order used)

1. Capture RGB888 frame from Picamera2 (sized to WIDTH×HEIGHT).
2. Grayscale → CLAHE → Gaussian blur (local contrast, denoise).
3. Canny (auto thresholds from image median) → Closing → Dilation (stable, connected borders).
4. Contours via findContours(edges, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE).
5. Sort by area → keep Top-K×2.
6. For each contour:
  1) minAreaRect → rotation-invariant width/height, aspect, area
  2) extent and solidity to reject grout lines, cables, text blobs
  3) build convex hull, then adaptive approxPolyDP until 4 points.
  4) Angle sanity check (suppress highly skewed quads).
  5) Score by size × filledness × rectangularity; collect candidates.
  6) If multiple strong candidates: merge their contours, hull again, re-approx to 4 points.
  7) Order corners (TL, TR, BR, BL) → compute homography → warpPerspective.
  8) OSD orientation correction (with short history) to avoid flipping; optional OCR overlay for QA. 

## Development path

1. Edge-map first.
Validated Canny settings across all orientations (0–360°) and lighting on the actual tiled floor. Added closing + dilation to keep the border continuous when the page was diagonal. Alternatives (Sobel/Laplacian) were noisier and less selective. 

<img width="1191" height="892" alt="image" src="https://github.com/user-attachments/assets/d65dea41-83f4-45f8-b6d6-d0af75d93a44" />


2. Full contour map.
Ensured the document outline appears reliably by switching to RETR_EXTERNAL to ignore internal text contours; this eliminated many false positives from small text blobs. 

<img width="1187" height="888" alt="image" src="https://github.com/user-attachments/assets/0ae2e64e-781a-4fa7-97c7-4ee2f05eb127" />


3. Ranking the candidates.
Verified that the true page contour lands in the top few by area in all scenes; used Top-K truncation for speed.

<img width="1190" height="891" alt="image" src="https://github.com/user-attachments/assets/6f782e2c-0814-403c-805c-df4efddae934" />


4. Selection criteria (narrowing down).
Rejected non-pages using:

5. Too small/large (minAreaRect width/height limits relative to image size),

6. Wrong aspect (outside 0.8–1.5),

7. Low extent (thin grout/cables) or low solidity (text clusters / perforated shapes),

8. Non-rectangular angles (min/max internal angles outside [50°, 130°]).
This mix kept the page and dropped: text, tile seams, cables, shadows, irregular blobs. 

9. True polygon, not a mere box.
Early versions used boxPoints(minAreaRect); replaced by convex hull + adaptive approxPolyDP to recover the real 4 corners, even when one edge was ragged/low-contrast. 

10. Exact warp & sizing.
Switched to averaging opposite sides to estimate target w×h and then getPerspectiveTransform → warpPerspective for a precise, non-stretched scan. 

11. Orientation stabilization.
Added Tesseract OSD and a small majority window to prevent flip-flop near 90°/180° ambiguities; kept it optional so OCR costs are paid only when needed. 

<img width="1500" height="910" alt="image" src="https://github.com/user-attachments/assets/e7fe9756-278c-49d8-a5c9-54963d5fc71a" />

<img width="1393" height="896" alt="image" src="https://github.com/user-attachments/assets/1449ab19-5e47-4b9f-b4c3-c784501a99c3" />

<img width="1262" height="897" alt="image" src="https://github.com/user-attachments/assets/b55a7519-bea3-458f-963a-8c9373249cd1" />

<img width="1702" height="622" alt="image" src="https://github.com/user-attachments/assets/5dea372c-f3e8-41f0-b76f-6db249a48bdd" />

<img width="1706" height="631" alt="image" src="https://github.com/user-attachments/assets/40a46a99-8a29-4456-908f-a5dce23d16d2" />

<img width="1682" height="691" alt="image" src="https://github.com/user-attachments/assets/bff321e8-4d13-4bb3-8eb0-f4ee6280bae4" />

<img width="1450" height="897" alt="image" src="https://github.com/user-attachments/assets/e04753fd-7459-4c92-bc8a-7dc857385ec7" />

<img width="1450" height="863" alt="image" src="https://github.com/user-attachments/assets/9e32edf3-dccd-4821-8d2e-1a7775c5d73c" />

<img width="1505" height="893" alt="image" src="https://github.com/user-attachments/assets/0e4b8b30-b8ee-44b9-9a3d-09c87d65ca9f" />

## What this setup filters out (by design)

1. Small text contours inside the page (ignored by RETR_EXTERNAL and later by solidity/extent thresholds).
2. Tile grout / cables / thin edges (low extent) and non-solid blobs (low solidity).
3. Highly skewed quads (angle sanity).
4. Too small or too large objects (relative side-length gates from minAreaRect). 

## Why not deep learning for page detection?

 - Lightweight, classic CV runs real-time on RPi without accelerators and needs no dataset/training.
 - DL alternatives (e.g., corner detectors, segmentation models) add compute and deployment overhead—unnecessary for a single, well-structured class like documents in controlled distance (≤1 m).

## How to run

Hardware: Raspberry Pi 5 + IMX500 (Picamera2), reasonably lit indoor scene.
Software: Python 3, OpenCV 4+, pytesseract (Tesseract 4/5), Picamera2.
Picamera2 manual (capture/format notes): RGB888 works well with OpenCV. 
Press q to quit. Windows show: All Contour Feed, topk Contour Feed, few Contour Feed, selected Contour Feed, scan, and optional OCR Annotated.

## Tuning knobs (mapped to code constants)

1. CLAHE: CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE
2. Canny: auto from median; adjust if needed via constants around aperture/L2 (CANNY_APERTURE_SIZE, CANNY_L2GRADIENT). 
3. Morphology: MORPH_KSIZE, MORPH_CLOSE_ITER, DILATE_ITER (join broken borders). 
4. Shortlisting: MIN_AREA, side-length bounds (MIN_SIDE_FRAC, MAX_SIDE_FRAC), ASPECT_MIN/MAX, EXTENT_MIN, SOLIDITY_MIN. 
5. Approximation: EPSILON, EPS_ADAPT_START, EPS_ADAPT_MULTS (tighter → more corners; looser → smoother). 
6. Merge policy: MERGE_TOP_REL_SCORE, MERGE_EPS_MULTS
7. OSD: OSD_CONF_MIN, OSD_HISTORY_LEN (raise if you still see occasional flips). 

## Summary

This pipeline purposely favors classic, lightweight vision that is:
1) rotation-agnostic (min-area rect gates + hull-based quad),
2) robust to indoor lighting (CLAHE + Canny + morphology),
3) fast on Raspberry Pi (external contours, simple shape features), and
4) stable in orientation (OSD + debounce).
