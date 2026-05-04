# RESAM (Radar-Efficient SAM)

## Overview
**RESAM (Radar-Efficient SAM)** is an end-to-end machine learning pipeline and annotation suite designed specifically for the segmentation of ships and unknown maritime targets from W-radar images.

Currently, the project leverages a state-of-the-art Vision Transformer (**Mask2Former** with a Swin-base backbone) for robust instance and semantic segmentation. In future releases, the architecture will be upgraded to implement a specialized, highly efficient version of the **Segment Anything Model 2 (SAM 2)** tailored for radar imagery. Furthermore, a dedicated object detection pipeline using **YOLO26** is planned specifically for elements in the dataset annotated with bounding boxes.

---

## Key Features

### 1. Advanced Radar Data Labeler (`radar_labeler.py`)
A custom-built, Tkinter-based GUI tailored for annotating W-radar images with both bounding boxes and precise polygon masks.
* **Hybrid Annotation:** Seamlessly switch between BBox and Mask modes.
* **Dynamic UI:** Includes an interactive categories legend, an image selection dropdown, a scrollable sidebar to manage/delete current annotations, and real-time mouse-wheel zooming.
* **Automated OCR Integration:** Extracts critical navigational data (Heading, Radar Range, Own Ship GPS coordinates, Cursor GPS coordinates, and UTC Time) directly from the radar image display using Tesseract OCR, employing a 2x Lanczos upscale to accurately resolve small text.
* **Radar Center Detection:** Dynamically detects whether the radar origin is "Centered" or "Off-Centered" based on OCR, or allows manual selection via a dropdown.
* **Log & AIS Synchronization:** Intelligently syncs OCR-extracted timestamps and coordinates with raw `.log` files (using a floating time window) to map historical AIS ship data and "Log Truth" GPS coordinates directly onto the radar canvas.
* **Instant Geographic Probing:** Right-clicking anywhere on the radar canvas instantly calculates and displays the exact Latitude/Longitude (in Degrees Decimal Minutes) of that pixel, automatically calibrated by the detected radar range.
* **Standardized Output:** Saves annotations in a structured JSON format compatible with modern segmentation pipelines.

### 2. Deep Learning Pipeline (`radar_detector.ipynb`)
A complete Jupyter Notebook workflow for data preparation, training, and evaluation.
* **Exploratory Data Analysis (EDA):** Visualizes class distributions and annotation types (BBox vs. Segmentation).
* **Robust Splitting:** Merges datasets from multiple folders and generates reproducible Train/Val/Test splits using 5-Fold Cross Validation via a global JSON map.
* **Custom Dataset Handling:** Automatically rasterizes polygons and bounding boxes into semantic masks. Implements a dynamic, resolution-agnostic crop that slices a perfect square from the left side of the image, paired with a dynamic circular mask to black out UI overlays and corners during training.
* **Transformer Training:** Fine-tunes `Mask2FormerForUniversalSegmentation` using PyTorch Automatic Mixed Precision (AMP), gradient clipping, and a custom fast binary foreground IoU metric. Features both a 5-Fold CV training loop and a Final Production model training run.
* **Inference & Evaluation:** Features a dynamic mapping engine to test various logical class groupings (e.g., merging "Unknown TGT" and "AIS Lost" into "Ship"). Calculates robust metrics (IoU, Precision, Recall, F1-Score) per class and overall, automatically visualizing and sorting the "Best" and "Worst" inference cases.

---

## Target Classes
The system is currently configured to detect and segment the following maritime categories:
1. `Ship`
2. `Noise`
3. `Unknown TGT`
4. `Coast/Port`
5. `My Ship`
6. `AIS Lost`

---

## Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies:

pip install torch torchvision transformers opencv-python scikit-learn matplotlib seaborn tqdm Pillow pytesseract pyais

*(Note: Tkinter is usually included with standard Python installations. If you are on Linux, you may need to install it via your package manager, e.g., `sudo apt-get install python3-tk`. You will also need Tesseract OCR installed on your system for the labeler's text extraction to function).*

---

## Usage Guide

### Part 1: Annotating Data
1. Prepare your dataset folder (e.g., `RealImages/`) containing your `.png` or `.jpg` radar images and their associated `_log` folders.
2. Inside that folder, create a `categories.txt` file using the `name = ID` format:
    Ship = 1
    Noise = 2
    Unknown TGT = 3
    Coast/Port = 4
    My Ship = 5
    AIS Lost = 6
3. Run the labeler:
    python radar_labeler.py
4. **Controls:**
    * **Load Folder:** Select your dataset directory.
    * **Image / Center Dropdowns:** Jump to specific images or manually set the radar origin.
    * **Mouse Wheel:** Zoom in and out.
    * **Left Click & Drag:** Draw Bounding Boxes (in BBox mode).
    * **Left Click:** Place polygon points (in Mask mode). Click near the start point to auto-close.
    * **Right Click / Enter:** Close the current polygon manually, or probe Geographic Coordinates on the canvas.
    * **A / D:** Previous / Next image.
    * **S:** Save JSON annotations.
    * **L:** Clear the last drawn point/shape.

### Part 2: Training and Inference
Open `radar_detector.ipynb` in Jupyter Notebook or JupyterLab. The notebook is divided into sequential execution blocks:

1.  **Dataset Analysis:** Run the first cell to generate a bar chart of class distributions and a pie chart of annotation types.
2.  **Dataset Splitting:** Run the `merge_and_split_datasets` function to aggregate all folders and partition the data into a 20% hold-out test set and a 5-fold CV TrainVal setup (saved to `MasterSplits/merged_dataset_splits.json`).
3.  **Model Training:**
    * Run the 5-Fold Cross Validation loop to evaluate model stability and hyperparameters.
    * Run the `train_final_production_model` function to train on 100% of the TrainVal data, saving the output as `vit_wradar_FINAL_PRODUCTION.pth`.
4.  **Inference & Evaluation:**
    * Select your desired logical class mapping (`original`, `mapping_1`, or `mapping_2`).
    * The script evaluates the production model on the held-out test set.
    * Outputs a detailed global metrics report (IoU, Precision, Recall) and saves sorted visualizations of the best and worst predictions.

---

## Future Roadmap

### 1. Integration of SAM 2
In upcoming iterations, RESAM will integrate the **Segment Anything Model 2 (SAM 2)**.
* **SAM Auto-Labeling:** Integrating SAM 2 into the Tkinter GUI to allow click-based auto-segmentation of radar targets, drastically reducing manual polygon drawing time.
* **Advanced Evaluation:** Utilizing SAM 2's prompt-based architecture (using existing bounding box annotations as prompts) to benchmark its pure boundary-segmentation accuracy against the fine-tuned Mask2Former model.

### 2. YOLO26 Bounding Box Detection
To optimize performance on specific dataset subsets, a dedicated **YOLO26** training pipeline will be introduced. This pipeline will exclusively target elements in the dataset annotated with bounding boxes. By leveraging YOLO26's native NMS-free end-to-end architecture and edge-optimized inference capabilities, this will provide a high-speed, detection-focused alternative to the Mask2Former segmentation approach.