<img width="781" height="441" alt="down_scale_00003_" src="https://github.com/user-attachments/assets/83817733-0a71-4668-a4a0-09621e8e9569" /># Edge Vision and AI Inference

This repo contains my experiments with SBC and SOCs like the Rapsberry PI 5 and Jetson Orin Nano 

The mini-projects present in this repo are - 
1. Real time scanning of document placed at arbitraty location, arbitrary orientation from a video feed and extracting text from it using Raspberry Pi 5, 5 MP camera and RPI AI camera
 - This involves use of edge detection and image processing algorithms like Sobel, Canny and CLAHE
 - Performing contour extraction and selection using OpenCV apis
 - Transforming and warping the extracted contour
 - Extracting text from warpped contour using tesseract

  <img width="750" height="455" alt="down_scale_00002_" src="https://github.com/user-attachments/assets/07cfe0d1-062a-4f37-9e41-29ed2d0c44df" />
   
2. Near real time Automatic brand detection and counting using template images of various brands
 - using 5 MP RPI Camera and RPI AI Camera (Sony IMX500)
 - SIFT for local features Chosen
 - FLANN approximate nearest neighbor search Chosen
 - Cluster-then-homography
 - Axis-aligned rectangles 

  <img width="781" height="441" alt="down_scale_00003_" src="https://github.com/user-attachments/assets/eb4e5c13-cf4a-4b6a-8112-5d09f2cfc7ac" />

3. Real time depth estimation
 - Monocular depth pipeline using a Raspberry Pi 5, IMX500 camera, and a lightweight depth model.
 - Publishes normalized depth, a colormap preview, camera intrinsics, and an optional metric mapping.
 - Strictly synchronized RGB + depth display and an external publisher for the 3D viewer.
 - Stereo depth pipeline using an OAK-D Lite.
 - Runs hardware stereo to produce metric depth in millimeters, optionally aligned to RGB.
 - Publishes depth at its native size with intrinsics for that exact size.
 - Unified live 3D point-cloud viewer (Qt + PyQtGraph OpenGL).

   <img width="871" height="476" alt="down_scale_00004_" src="https://github.com/user-attachments/assets/b1efd3e8-49fa-4c18-af42-dec8c278ede3" />

   <img width="866" height="441" alt="down_scale_00005_" src="https://github.com/user-attachments/assets/c45077a7-1ae6-42db-94c2-decef4fdca44" />

   <img width="866" height="441" alt="down_scale_00005_" src="https://github.com/user-attachments/assets/8d1e007e-16d6-4d9a-99ab-9a334e82f5d7" />

4.  Near real time image classification using transfer learning to fine tune lightweight MobileNetV2 on openimages and perform inference with TFLite (raspberry pi)
5.  Near real time object detection using yolov5n and yolov5s and classifying using fine tuned MobileNetV2 from previous project (raspberry pi)
6.  Near real time edge segmentation using pretrained LRASPP MobileNetV3 together with adding the detector and classifier from the last two projects (raspberry pi)
7.  Eye-blink detection using openCV (raspberry pi)
8.  Inferencing LLM models using ollama CLI
9.  Inferencing LLM models using ollama server and Open Web UI
10. Setting up vision inferencing Stable Diffusion UI
11. Setting up and creating workflows using ComfyUI to create deterministic workflows to perform vision tasks with numerous model checkpoints, Lora adaptors, embeddings, etc for image generation
12. Building a generic coding agent using the ollama backend
 - Has access to multiple 7B to 8B LLM coding specific model
 - Uses RAG from local knowledge base created out of vector embeddings of documentations of Python, C, C++, Java, JS, etc
 - Creates github repo,
 - Plans directory structure
 - Plans the complete coding task
 - Creates tests based on the requirements
 - Generates the code
 - Tests the code
 - Performs corrections in the code and the tests
 - Generates Readme and comments the code
 - personal github repo of the coding agent - [astro_repos](https://github.com/kanthrishav/astro_repos)

