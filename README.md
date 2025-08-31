# Edge Vision and AI Inference

This repo contains my experiments with SBC and SOCs like the Rapsberry PI 5 and Jetson Orin Nano 

The mini-projects present in this repo are - 
1. Real time scanning of document placed at arbitraty location, arbitrary orientation from a video feed and extracting text from it using Raspberry Pi 5, 5 MP camera and RPI AI camera
 - This involves use of edge detection and image processing algorithms like Sobel, Canny and CLAHE
 - Performing contour extraction and selection using OpenCV apis
 - Transforming and warping the extracted contour
 - Extracting text from warpped contour using tesseract

  <img width="1500" height="910" alt="image" src="https://github.com/user-attachments/assets/a83ffe0a-a7a7-4d09-b23a-2644eb929057" />

   
2. Near real time Automatic brand detection and counting using template images of various brands
 - using 5 MP RPI Camera and RPI AI Camera (Sony IMX500)
 - SIFT for local features Chosen
 - FLANN approximate nearest neighbor search Chosen
 - Cluster-then-homography
 - Axis-aligned rectangles 

  <img width="1562" height="882" alt="image" src="https://github.com/user-attachments/assets/dbe8da33-9b3b-4c3b-b7ec-70f667f45d80" />

3. Real time depth estimation
 - Monocular depth pipeline using a Raspberry Pi 5, IMX500 camera, and a lightweight depth model.
 - Publishes normalized depth, a colormap preview, camera intrinsics, and an optional metric mapping.
 - Strictly synchronized RGB + depth display and an external publisher for the 3D viewer.
 - Stereo depth pipeline using an OAK-D Lite.
 - Runs hardware stereo to produce metric depth in millimeters, optionally aligned to RGB.
 - Publishes depth at its native size with intrinsics for that exact size.
 - Unified live 3D point-cloud viewer (Qt + PyQtGraph OpenGL).

  <img width="1742" height="951" alt="image" src="https://github.com/user-attachments/assets/45eababb-5fd8-412c-906f-d76cfe87d549" />

  <img width="1733" height="882" alt="image" src="https://github.com/user-attachments/assets/a14d964a-b1b1-4051-80b1-b4e28e93d692" />

  <img width="1701" height="835" alt="image" src="https://github.com/user-attachments/assets/c31e7fd6-2af9-42db-ad68-35ae5630aaaa" />

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

