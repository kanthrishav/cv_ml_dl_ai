# Edge Vision and AI Inference

This repo contains my experiments with SBC and SOCs like the Rapsberry PI 5 and Jetson Orin Nano 

The mini-projects present in this repo are - 

## 1. Real time scanning of document placed at arbitraty location, arbitrary orientation from a video feed and extracting text from it using Raspberry Pi 5, 5 MP camera and RPI AI camera
 - This involves use of edge detection and image processing algorithms like Sobel, Canny and CLAHE
 - Performing contour extraction and selection using OpenCV apis
 - Transforming and warping the extracted contour
 - Extracting text from warpped contour using tesseract

    <img width="750" height="455" alt="down_scale_00002_" src="https://github.com/user-attachments/assets/07cfe0d1-062a-4f37-9e41-29ed2d0c44df" />

******************************************************************************************************************************************************************************************************************

   
## 2. Near real time Automatic brand detection and counting using template images of various brands
 - using 5 MP RPI Camera and RPI AI Camera (Sony IMX500)
 - SIFT for local features Chosen
 - FLANN approximate nearest neighbor search Chosen
 - Cluster-then-homography
 - Axis-aligned rectangles 

    <img width="782" height="372" alt="image" src="https://github.com/user-attachments/assets/ef036812-165c-4438-b8d1-5ad04dd73408" />

******************************************************************************************************************************************************************************************************************


## 3. Real time depth estimation
 - Monocular depth pipeline using a Raspberry Pi 5, IMX500 camera, and a lightweight depth model.
 - Publishes normalized depth, a colormap preview, camera intrinsics, and an optional metric mapping.
 - Strictly synchronized RGB + depth display and an external publisher for the 3D viewer.
 - Stereo depth pipeline using an OAK-D Lite.
 - Runs hardware stereo to produce metric depth in millimeters, optionally aligned to RGB.
 - Publishes depth at its native size with intrinsics for that exact size.
 - Unified live 3D point-cloud viewer (Qt + PyQtGraph OpenGL).

<table><tr><td><img width="871" height="476" alt="down_scale_00004_" src="https://github.com/user-attachments/assets/b1efd3e8-49fa-4c18-af42-dec8c278ede3" /></td><td><img width="866" height="441" alt="down_scale_00005_" src="https://github.com/user-attachments/assets/c45077a7-1ae6-42db-94c2-decef4fdca44" /></td><td><img width="850" height="418" alt="down_scale_00001_" src="https://github.com/user-attachments/assets/84ae3219-13c8-4778-9ebb-2f86dce7cb05" /></td></tr></table>


******************************************************************************************************************************************************************************************************************


## 4.  Inferencing LLM models using ollama CLI on Jetson Orin Nano (running 100% locally)

<img width="1200" height="630" alt="image" src="https://github.com/user-attachments/assets/8574c0e3-fe2a-4732-81ca-d43843bba51e" />


https://github.com/user-attachments/assets/920d8ddd-b529-4203-ada7-4a97d128e0d6

******************************************************************************************************************************************************************************************************************

  
## 5.  Inferencing LLM models using ollama server and Open Web UI on Jetson Orin Nano (running 100% locally)

<img width="1916" height="981" alt="image" src="https://github.com/user-attachments/assets/0d4c10da-235c-4296-b151-12c53f2995b6" />

https://github.com/user-attachments/assets/fa38dda3-ef41-4131-b1f1-dc45c73824b3

******************************************************************************************************************************************************************************************************************

## 6. Setting up and creating workflows using ComfyUI to create deterministic workflows to perform vision tasks with numerous model checkpoints, Lora adaptors, embeddings, etc for image generation (running 100% locally)

Example of converting a random empty living room into an Indian, a Parisian and a Japanese living room

<img width="1917" height="975" alt="image" src="https://github.com/user-attachments/assets/128bdb5a-fc98-4a00-b74a-045aee4ff844" />

<table><tr><td><img width="400" height="260" alt="indian" src="https://github.com/user-attachments/assets/e1eb60c8-729f-42e7-9f7e-707a4dbda6e8" /></td><td><img width="400" height="260" alt="japanese" src="https://github.com/user-attachments/assets/d1e0fbb1-4f45-4ed5-9f6f-ccbec0c368f8" /></td><td><img width="400" height="260" alt="parisian" src="https://github.com/user-attachments/assets/f0d04e03-5385-4aec-9741-e52cc2704087" /></td></tr></table>


https://github.com/user-attachments/assets/d3562c5a-5be4-4555-8d1e-7814312debb1

******************************************************************************************************************************************************************************************************************

## 7. Setting up vision inferencing Stable Diffusion UI

<img width="1917" height="973" alt="image" src="https://github.com/user-attachments/assets/bf9d8635-bba0-47f0-bbaf-f4f15b688de6" />

******************************************************************************************************************************************************************************************************************


## 8.  RAG based Chatbot with ollama backend

<img width="1917" height="981" alt="image" src="https://github.com/user-attachments/assets/fc099b69-2493-4331-8b80-e2b2cba86c69" />


https://github.com/user-attachments/assets/1fb88318-4097-4944-8b43-df97fe9744c7


******************************************************************************************************************************************************************************************************************


## 9.  Near real time image classification using transfer learning to fine tune lightweight MobileNetV2 on openimages and perform inference with TFLite (raspberry pi)

******************************************************************************************************************************************************************************************************************


## 10.  Near real time object detection using yolov5n and yolov5s and classifying using fine tuned MobileNetV2 from previous project (raspberry pi)

******************************************************************************************************************************************************************************************************************


## 11.  Near real time edge segmentation using pretrained LRASPP MobileNetV3 together with adding the detector and classifier from the last two projects (raspberry pi)

******************************************************************************************************************************************************************************************************************


## 12.  Eye-blink detection using openCV (raspberry pi)

******************************************************************************************************************************************************************************************************************


## 13. Building a generic coding agent using the ollama backend
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
 - personal github repo of the coding agent ( everything present in this repo is generated by the coding agent completely, with 100% on-device local inference) - [astro_repos](https://github.com/kanthrishav/astro_repos)

