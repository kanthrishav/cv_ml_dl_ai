# Brand Localizer on Raspberry Pi 5 with IMX500

<img width="774" height="436" alt="down_scale_00019_" src="https://github.com/user-attachments/assets/0a4e32f8-042f-41fb-bf0c-4fe05f485b7b" />

<img width="784" height="442" alt="down_scale_00020_" src="https://github.com/user-attachments/assets/61bf0576-d8ae-4937-bafb-0671b4381c6c" />

<img width="781" height="441" alt="down_scale_00021_" src="https://github.com/user-attachments/assets/b1c0ffb5-3599-48ae-934a-ee8f778fd9fc" />

A compact, template-based logo detector that counts multiple brands in real time using classical CV on the Raspberry Pi. This README documents the approach, the variants, why each technique was chosen, and how the system evolved.

## Summary

This mini-project detects and counts branded logos in a live video stream captured from a Raspberry Pi 5 with an IMX500 camera. It uses classical feature-based vision to remain lightweight and to avoid training data requirements. Two tuned variants are included:

Close-range variant for approximately 90 cm to 100 cm subject distance. Higher recall at short range. Lower frame rate.
Far-range variant for approximately 100 cm to 200 cm subject distance. More tolerant to smaller logos, with a lower frame rate.
Both variants:
Accept multiple templates, one image per brand.
Report per-brand counts with stable, axis-aligned rectangles.
Remove duplicate detections and smooth box motion over time.

## Hardware and Physical Setup

1. Host: Raspberry Pi 5 with 8 GB RAM.
2. Camera: Raspberry Pi AI Camera IMX500 (possible with any RPI camera)
3. Resolution: 1280 by 720 pixels for detection and display.
4. Target distances:
Close-range variant tuned for approximately 90 cm to 100 cm.
Far-range variant tuned for approximately 100 cm to 200 cm.
5. Focus setting: user set near approximately 80 cm.
6. Lighting: consistent indoor lighting recommended to maximize keypoint quality.
Physical parameters affect performance more than algorithmic settings. Small logos, long distances, glare, motion blur, and large perspective skew reduce the number and quality of feature matches.

## Project Structure

1. Templates directory: templates/
Contains one image per brand.
Images can be PNG or JPG.
2. Detector scripts:
Close-range script: tuned for 90 cm to 100 cm.
Far-range script: tuned for 100 cm to 200 cm.
3. No model files: classical CV only, no learned weights.

## How It Works

The pipeline is the same for both variants; only thresholds and SIFT configuration are tuned.
Asynchronous capture
A camera thread streams RGB frames at 1280 by 720 using Picamera2.
The main thread reads the latest frame without blocking.
1. Template preparation
On startup, each template is resized to fit the working resolution.
SIFT features and descriptors are computed once per template.
A FLANN matcher is created per template for fast descriptor search.

2. Frame preprocessing
Convert the frame to BGR for OpenCV, then to gray.
Apply CLAHE to improve local contrast for feature detection.
Compute SIFT features and descriptors for the scene.

3. Matching
Mutual nearest neighbor plus Lowe ratio test is applied:
template to scene with k-NN equals 2 for the ratio test,
scene to template with k-NN equals 1 to enforce reciprocity.

4. Spatial clustering
Draw small discs at matched scene keypoint locations and perform a morphological closing.
Extract connected components to obtain clusters of matches corresponding to potential object instances.

5. Geometric verification
For each cluster, estimate a homography with RANSAC.
Accept only if:
Minimum inlier count and inlier ratio are satisfied.
Mean reprojection error is below a quality threshold.
The projected template quad is sane in size and within the image.

6. Rectangle synthesis
Convert the projected quad to an axis-aligned bounding rectangle.
Reject rectangles with aspect ratios far from the template.

7. De-duplication
Apply IoU-based NMS and center-distance merging to collapse near-duplicates into a single detection.

8. Lightweight tracking
Associate boxes across frames using IoU.
Smooth box position and size with an exponential moving average.
Guard against sudden area jumps and drop stale tracks.

9. Counting and display
Draw green rectangles and per-brand counts.
Overlay a smoothed FPS estimate.

## Techniques Used, Why They Were Chosen, and Alternatives
1. SIFT for local features
Chosen because it is robust to scale, rotation, and some illumination changes. Works out of the box with few constraints and no training data.
Alternatives:
ORB or AKAZE: faster but noticeably less stable for small, low-contrast text logos at distance on 1280 by 720 frames. Produced more flicker and missed instances.
SURF: patented historically and not always available in default builds.
Deep detectors (for example small YOLO variants): would require dataset preparation and training, plus integration overhead. Excellent once trained, but not aligned with the zero-training goal of this mini project.

2. FLANN approximate nearest neighbor search
Chosen because it scales well for descriptor sets and is supported natively in OpenCV.
Alternative: brute-force matching with L2.

3. Cluster-then-homography
Reason: allows multiple instances of the same brand in one frame and prunes spurious matches before expensive geometry.
Alternative: one global homography per brand would merge instances and fail with multiple objects.

4. Axis-aligned rectangles
Reason: more stable, easier to read, and simpler to deduplicate.
Alternative: drawing raw perspective quads leads to visual jitter and odd shapes.

5. Lightweight tracker
Reason: smooths boxes and prevents duplicate flicker without the overhead of a full multi-object tracker.


## Development Sequence

1. Baseline capture and template matching
Implemented Picamera2 acquisition and naive feature matching against each template.
2. Introduce ratio test and mutual matches
Reduced false positives by enforcing symmetric nearest neighbors plus Lowe ratio.
3. Add spatial clustering
Grouped matches to support multiple instances per brand and to isolate homography fits.
4. Integrate homography with quality gates
Added RANSAC, inlier ratio, reprojection error, quad sanity, and aspect-ratio checks.
5. Convert quads to rectangles
Improved visual stability and simplified counting.
6. Apply NMS and center merge
Eliminated duplicates from dense keypoint clusters.
7. Add lightweight tracking
Introduced IoU association, EMA smoothing, area jump guard, and miss TTL for continuity.
8. Mid-range tuning
Adjusted SIFT parameters, clustering radius, and inlier floors to improve recognition at 100 to 200 cm, at the cost of throughput.
9. Code cleanup and documentation
Standardized naming, constants, and comments for maintainability.

## How To Run

 - Place brand images in templates. One brand per file. File name becomes the label.
 - Choose the variant:
 - brand_localizer_close.py for 90 to 100 cm.
 - brand_localizer_mid.py for 100 to 200 cm.
 - Run the script. Press the q key to exit.
 - Optional: keep the template preview windows open to inspect detected keypoints.

## Adding New Brands

 - Capture a clear, fronto-parallel image of the brand mark.
 - Save it in templates/ using a concise name such as amazon.png.
 - The loader rescales the template to fit the detection resolution and computes features automatically.

## Tuning Knobs
1. SIFT_FEATURES
Increase for more keypoints in low-texture scenes. Larger values reduce fps.
2. RATIO_TEST
Lower values are stricter and may drop true matches in low-contrast regions.
3. FLANN_CHECKS
Higher values give better nearest neighbors but reduce fps.
4. MIN_CLUSTER_MATCH and per-template adaptive minimum
Raise to suppress weak clusters; lower to find smaller logos.
5. RANSAC_THRESH, MIN_INLIERS_ABS, INLIER_RATIO_MIN, MAX_REPROJ_ERR
Tighten to reduce false positives; loosen to catch difficult cases.
6. IOU_NMS_THRESH and CENTER_MERGE_FRAC
Adjust de-duplication behavior.
7. SMOOTH_ALPHA, IOU_ASSOC_THRESH, MISS_TTL_FRAMES, SIZE_JUMP_MAX
Control tracker responsiveness and stability.

## Known Limitations

Throughput
Classical SIFT on CPU is compute-bound. Expect low single-digit fps for multiple templates at 1280×720.

Distance scaling
Very small logos at long distances produce few reliable keypoints, reducing recall.

Strong motion blur or low light
Reduces feature repeatability and homography success.

Highly repetitive textures
May create many ambiguous matches; gates mitigate but cannot fully eliminate this in extreme cases.

## Future Directions

On-sensor neural inference
Port a compact detector to the IMX500 to pre-select regions, then verify with the homography step.

Hybrid pipeline
Use a fast learned text or logo proposal network to seed the SIFT pipeline for verification and counting.

Scale-space heuristics
Dynamic SIFT parameterization based on measured target size in the scene to balance recall and fps.

Template set management
Auto-prune or cluster templates to cap per-frame matcher workload.

## Conclusion

This mini project provides a maintainable, well-documented classical-CV solution for multi-brand detection under two operating regimes. It emphasizes transparent geometry, explainable gates, and deterministic behavior while keeping the code approachable for future extension to on-sensor neural proposals or hybrid detection-verification workflows.






