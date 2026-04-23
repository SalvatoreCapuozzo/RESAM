# RESAM (Radar-Efficient SAM)

## Overview
**RESAM (Radar-Efficient SAM)** is an end-to-end machine learning pipeline and annotation suite designed specifically for the segmentation of ships and unknown maritime targets from W-radar images. 

Currently, the project leverages a state-of-the-art Vision Transformer (**Mask2Former** with a Swin-base backbone) for robust instance and semantic segmentation. In future releases, the architecture will be upgraded to implement a specialized, highly efficient version of the **Segment Anything Model (SAM)** tailored for radar imagery.

---

## Key Features

### 1. Advanced Radar Data Labeler (`radar_labeler.py`)
A custom-built, Tkinter-based GUI tailored for annotating W-radar images with both bounding boxes and precise polygon masks.
* **Hybrid Annotation:** Seamlessly switch between BBox and Mask modes.
* **Dynamic UI:** Includes an interactive categories legend, a scrollable sidebar to manage/delete current annotations, and real-time mouse-wheel zooming.
* **Standardized Output:** Saves annotations in a structured JSON format compatible with modern segmentation pipelines.
* **Data-Driven:** Automatically parses class mappings from a local `categories.txt` file.

### 2. Deep Learning Pipeline (`radar_detector.ipynb`)
A complete Jupyter Notebook workflow for data preparation, training, and evaluation.
* **Exploratory Data Analysis (EDA):** Visualizes class distributions and annotation types (BBox vs. Segmentation).
* **Robust Splitting:** Generates reproducible Train/Val/Test splits using 5-Fold Cross Validation.
* **Custom Dataset Handling:** Automatically rasterizes polygons and bounding boxes into semantic masks, with built-in UI cropping to ignore radar screen overlays.
* **Transformer Training:** Fine-tunes `Mask2FormerForUniversalSegmentation` using PyTorch Automatic Mixed Precision (AMP), gradient clipping, and a custom fast binary foreground IoU metric.
* **Inference & Evaluation:** Calculates overall and per-class mean Intersection over Union (mIoU) and outputs side-by-side comparative visualizations (Original vs. Ground Truth vs. Prediction).

---

## Target Classes
The system is currently configured to detect and segment the following maritime categories:
1. `Ship`
2. `Noise`
3. `Unknown TGT`
4. `Coast/Port`
5. `My Ship`

---

## Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch torchvision transformers opencv-python scikit-learn matplotlib seaborn tqdm Pillow
```
*(Note: Tkinter is usually included with standard Python installations. If you are on Linux, you may need to install it via your package manager, e.g., `sudo apt-get install python3-tk`)*

---

## Usage Guide

### Part 1: Annotating Data
1. Prepare your dataset folder (e.g., `RealImages/`) containing your `.png` or `.jpg` radar images.
2. Inside that folder, create a `categories.txt` file using the `name = ID` format:
    ```text
    Ship = 1
    Noise = 2
    Unknown TGT = 3
    Coast/Port = 4
    My Ship = 5
    ```
3. Run the labeler:
    ```bash
    python radar_labeler.py
    ```
4. **Controls:**
    * **Load Folder:** Select your dataset directory.
    * **Mouse Wheel:** Zoom in and out.
    * **Left Click & Drag:** Draw Bounding Boxes (in BBox mode).
    * **Left Click:** Place polygon points (in Mask mode). Click near the start point to auto-close.
    * **Right Click / Enter:** Close the current polygon manually.
    * **A / D:** Previous / Next image.
    * **S:** Save JSON annotations.
    * **L:** Clear the last drawn point/shape.

### Part 2: Training and Inference
Open `radar_detector.ipynb` in Jupyter Notebook or JupyterLab. The notebook is divided into sequential execution blocks:

1.  **Dataset Analysis:** Run the first cell to generate a bar chart of class distributions and a pie chart of annotation types.
2.  **Dataset Splitting:** Run the splitting cells to partition the JSON files into a 5-fold cross-validation setup (saved to `output_splits/dataset_splits.json`).
3.  **Model Training:** * The training function `train_vit_segmentation` initializes the `Mask2Former` model.
    * It uses `Mask2FormerImageProcessor` to convert your custom JSON annotations into pixel-level semantic and instance masks.
    * Training utilizes AMP (Autocast + GradScaler) for memory-efficient GPU training.
    * The best model weights are automatically saved based on Validation IoU.
4.  **Inference & Evaluation:**
    * Loads the best saved `.pth` weights.
    * Runs inference on the held-out test set.
    * Prints a detailed Per-Class IoU report.
    * Generates side-by-side visualizations with automated legends in the `inference_results/` directory.

---

## Future Roadmap: Integration of SAM
The current implementation relies on Mask2Former. In upcoming iterations, RESAM will integrate the **Segment Anything Model (SAM)**. 
* **SAM Auto-Labeling:** Integrating SAM into the Tkinter GUI to allow click-based auto-segmentation of radar targets, drastically reducing manual polygon drawing time.
* **Efficient SAM Fine-tuning:** Adapting lightweight variants of SAM (e.g., MobileSAM or FastSAM) for real-time maritime radar inference, leveraging prompt-based segmentation (bounding box to mask).