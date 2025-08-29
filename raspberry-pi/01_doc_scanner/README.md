📄 Document Scanner (RPi 5 + Picamera2 + OpenCV) — README

Robust, orientation-agnostic document detection and scan warping on Raspberry Pi 5 (IMX500 + Picamera2), with optional OCR and auto-rotation.

⚙️ What this project does (at a glance)

Captures RGB888 frames from Picamera2 and pre-processes them for stable edges across lighting/orientation. 
Raspberry Pi Datasheets

Builds an edge map (Canny) and stabilizes borders via morphological closing + dilation. 
OpenCV Documentation
+3
OpenCV Documentation
+3
OpenCV Documentation
+3

Extracts external contours only, shortlists by rotation-invariant size & aspect using minAreaRect, and rejects text/grout via extent and solidity. 
OpenCV Documentation
+4
OpenCV Documentation
+4
OpenCV Documentation
+4

Forces a true 4-corner page polygon by approximating the convex hull (approxPolyDP, Douglas–Peucker). 
OpenCV Documentation

Computes a 4-point perspective transform and warps to a top-down scan. 
OpenCV Documentation

(Optional) Uses Tesseract OSD for orientation correction (0/90/180/270) with short history (debounce). 
PyPI
Tesseract OCR

🧠 Techniques, why they were chosen, and alternatives
1) Local contrast normalization (CLAHE) → Gaussian blur

Why: CLAHE boosts text/background separation under uneven indoor lighting; blur reduces high-frequency noise before Canny. 
OpenCV Documentation
+1

Alternatives:

Global histogram equalization (simpler, but can wash out highlights) 
OpenCV Documentation

Color-space thresholding (e.g., LAB ‘L’ channel) — workable but more scene-specific tuning.

Reason not used instead: CLAHE is more robust across floor colors/tiles and shadows than a global equalize or fixed thresholds. 
OpenCV Documentation

2) Canny edges + morphology (closing, dilation)

Why: Canny’s hysteresis gives clean, thin borders; closing seals gaps at steep rotations; dilation reconnects broken border segments. 
OpenCV Documentation
+2
OpenCV Documentation
+2

Alternatives:

Sobel/Scharr gradient magnitude + threshold (more tuning),

HED/learned edge detectors (heavy for RPi).

Reason not used instead: Canny + simple morphology hits the speed/quality sweet spot on Raspberry Pi. 
OpenCV Documentation

3) Contour retrieval (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

Why: Only the outermost shapes matter for a page; skipping child contours avoids selecting text inside the page. CHAIN_APPROX_SIMPLE reduces memory/CPU. 
OpenCV Documentation
Stack Overflow

Alternatives: RETR_TREE with hierarchy logic.

Reason not used: Full hierarchies are slower and unnecessary once we only want the page boundary. 
OpenCV Documentation

4) Rotation-invariant gating via minAreaRect, then extent & solidity

Why: minAreaRect gives width/height regardless of rotation; extent = area / (bbox area) filters long skinny grout lines; solidity = area / (hull area) rejects text clusters / non-solid shapes. 
OpenCV Documentation
+1

Alternatives:

Pure area threshold (fails on angled pages),

Hough lines + rectangle assembly (fragile with textured floors).

Reason not used: Min-area rectangle features are simpler and robust to rotation; extent/solidity come straight from classic contour features. 
OpenCV Documentation

5) Convex-hull → adaptive approxPolyDP to force 4 corners

Why: Pages are convex; approximating the hull avoids interior wiggles from text/texture. approxPolyDP (Douglas–Peucker) yields a clean quad when epsilon is adapted moderately. 
OpenCV Documentation

Alternatives:

boxPoints(minAreaRect) (fast but not the true page if one edge is curved/occluded),

Line-segment detectors (LSD/EDLines) + corner finding (heavier; parameter-sensitive).

Reason not used: Hull+DP is fast, stable, and tracks the real boundary better than a pure oriented box. 
OpenCV Documentation

6) 4-point perspective warp (getPerspectiveTransform → warpPerspective)

Why: Exact homography from ordered TL-TR-BR-BL points produces a metric-correct “scan”. 
OpenCV Documentation

Alternatives: Affine warp (insufficient for perspective foreshortening).

Reason not used: A4/sheet-like documents require a full projective transform. 
OpenCV Documentation

7) Orientation with Tesseract OSD (debounced majority)

Why: OSD returns discrete rotations (0/90/180/270) + confidence; a short history suppresses flip-flop near ambiguous angles. 
PyPI
Tesseract OCR

Alternatives: Hough-based baseline voting / gradient-energy ratios; good backups but OSD is language-aware and simple to integrate. 
PyImageSearch

🔁 Processing sequence (exact order used)

Capture RGB888 frame from Picamera2 (sized to WIDTH×HEIGHT). 
Raspberry Pi Datasheets

Grayscale → CLAHE → Gaussian blur (local contrast, denoise). 
OpenCV Documentation
+1

Canny (auto thresholds from image median) → Closing → Dilation (stable, connected borders). 
OpenCV Documentation
+1

Contours via findContours(edges, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE). 
OpenCV Documentation

Sort by area → keep Top-K×2.

For each contour:

minAreaRect → rotation-invariant width/height, aspect, area;

extent and solidity to reject grout lines, cables, text blobs;

build convex hull, then adaptive approxPolyDP until 4 points. 
OpenCV Documentation
+1

Angle sanity check (suppress highly skewed quads).

Score by size × filledness × rectangularity; collect candidates.

If multiple strong candidates: merge their contours, hull again, re-approx to 4 points.

Order corners (TL, TR, BR, BL) → compute homography → warpPerspective. 
OpenCV Documentation

OSD orientation correction (with short history) to avoid flipping; optional OCR overlay for QA. 
PyPI
Tesseract OCR

🧭 Development path (how we got here)

Edge-map first.
Validated Canny settings across all orientations (0–360°) and lighting on the actual tiled floor. Added closing + dilation to keep the border continuous when the page was diagonal. Alternatives (Sobel/Laplacian) were noisier and less selective. 
OpenCV Documentation
+1

Full contour map.
Ensured the document outline appears reliably by switching to RETR_EXTERNAL to ignore internal text contours; this eliminated many false positives from small text blobs. 
OpenCV Documentation

Ranking the candidates.
Verified that the true page contour lands in the top few by area in all scenes; used Top-K truncation for speed.

Selection criteria (narrowing down).
Rejected non-pages using:

Too small/large (minAreaRect width/height limits relative to image size),

Wrong aspect (outside 0.8–1.5),

Low extent (thin grout/cables) or low solidity (text clusters / perforated shapes),

Non-rectangular angles (min/max internal angles outside [50°, 130°]).
This mix kept the page and dropped: text, tile seams, cables, shadows, irregular blobs. 
OpenCV Documentation

True polygon, not a mere box.
Early versions used boxPoints(minAreaRect); replaced by convex hull + adaptive approxPolyDP to recover the real 4 corners, even when one edge was ragged/low-contrast. 
OpenCV Documentation

Exact warp & sizing.
Switched to averaging opposite sides to estimate target w×h and then getPerspectiveTransform → warpPerspective for a precise, non-stretched scan. 
OpenCV Documentation

Orientation stabilization.
Added Tesseract OSD and a small majority window to prevent flip-flop near 90°/180° ambiguities; kept it optional so OCR costs are paid only when needed. 
PyPI
Tesseract OCR

🧪 What this setup filters out (by design)

Small text contours inside the page (ignored by RETR_EXTERNAL and later by solidity/extent thresholds). 
OpenCV Documentation

Tile grout / cables / thin edges (low extent) and non-solid blobs (low solidity). 
OpenCV Documentation

Highly skewed quads (angle sanity).

Too small or too large objects (relative side-length gates from minAreaRect). 
OpenCV Documentation

🧾 Why not deep learning for page detection?

Lightweight, classic CV runs real-time on RPi without accelerators and needs no dataset/training.

DL alternatives (e.g., corner detectors, segmentation models) add compute and deployment overhead—unnecessary for a single, well-structured class like documents in controlled distance (≤1 m).

🛠️ How to run

Hardware: Raspberry Pi 5 + IMX500 (Picamera2), reasonably lit indoor scene.

Software: Python 3, OpenCV 4+, pytesseract (Tesseract 4/5), Picamera2.

Picamera2 manual (capture/format notes): RGB888 works well with OpenCV. 
Raspberry Pi Datasheets

Press q to quit. Windows show: All Contour Feed, topk Contour Feed, few Contour Feed, selected Contour Feed, scan, and optional OCR Annotated.

🔧 Tuning knobs (mapped to code constants)

CLAHE: CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE

Canny: auto from median; adjust if needed via constants around aperture/L2 (CANNY_APERTURE_SIZE, CANNY_L2GRADIENT). 
OpenCV Documentation

Morphology: MORPH_KSIZE, MORPH_CLOSE_ITER, DILATE_ITER (join broken borders). 
OpenCV Documentation

Shortlisting: MIN_AREA, side-length bounds (MIN_SIDE_FRAC, MAX_SIDE_FRAC), ASPECT_MIN/MAX, EXTENT_MIN, SOLIDITY_MIN. 
OpenCV Documentation

Approximation: EPSILON, EPS_ADAPT_START, EPS_ADAPT_MULTS (tighter → more corners; looser → smoother). 
OpenCV Documentation

Merge policy: MERGE_TOP_REL_SCORE, MERGE_EPS_MULTS

OSD: OSD_CONF_MIN, OSD_HISTORY_LEN (raise if you still see occasional flips). 
PyPI

📚 Key references

Canny (theory & parameters): OpenCV docs. 
OpenCV Documentation
+1

Morphology (closing, dilation): OpenCV tutorials. 
OpenCV Documentation
+1

Contours (retrieval modes, hierarchy, features): OpenCV tutorials. 
OpenCV Documentation
+2
OpenCV Documentation
+2

Min-area rectangle & boxPoints: OpenCV contour features. 
OpenCV Documentation

Polygon approximation (Douglas–Peucker): OpenCV approxPolyDP. 
OpenCV Documentation

Perspective warp: getPerspectiveTransform/warpPerspective. 
OpenCV Documentation

Tesseract OSD: PyPI / tessdoc usage notes. 
PyPI
Tesseract OCR

Picamera2 manual (capture formats, usage): Raspberry Pi datasheet. 
Raspberry Pi Datasheets

✅ Summary

This pipeline purposely favors classic, lightweight vision that is:

rotation-agnostic (min-area rect gates + hull-based quad),

robust to indoor lighting (CLAHE + Canny + morphology),

fast on Raspberry Pi (external contours, simple shape features), and

stable in orientation (OSD + debounce).
