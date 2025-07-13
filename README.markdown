# Computer Vision Labs

This repository contains a collection of computer vision lab experiments conducted as part of my coursework at the National University of Modern Languages, Islamabad, submitted on October 31, 2024, under the supervision of Mam Iqra Nasem. The labs explore fundamental and advanced computer vision techniques using OpenCV in Python, covering image processing, feature detection, object detection, and video manipulation. These labs build practical skills in analyzing and manipulating visual data, with applications in object recognition, edge detection, and real-time processing.

The project complements my prior coursework in deep learning (e.g., CNN-based classification) and speech processing (e.g., feature extraction), particularly for labs involving neural networks or contour-based detection.

## Lab Experiments

The `notebooks/` folder contains Python scripts or Jupyter notebooks for each lab, implementing computer vision techniques with OpenCV. The labs are summarized below, with their objectives and related files. **Note**: The content is shared with permission for educational purposes and has been summarized to exclude proprietary information.

| Lab | Title | Description | Related Files |
| --- | --- | --- | --- |
| 1 | Bitwise Operations | Applies bitwise operations (AND, OR, XOR, NOT) on images to manipulate pixel values for masking and overlaying. | bitwise_operations.py |
| 2 | Extracting Frames | Extracts frames from a video file and saves them as images for analysis or processing. | extract_frames.py |
| 3 | Color Detection Using Mask | Detects specific colors in images using HSV color space and masking techniques. | color_detection.py |
| 4 | Color Picker | Creates an interactive tool to select colors from an image and display their RGB/HSV values. | color_picker.py |
| 5 | Flip Operations | Performs horizontal, vertical, and combined flips on images to demonstrate spatial transformations. | flip_operations.py |
| 6 | Contour Functions | Identifies and draws contours in images to detect object boundaries. | contour_functions.py |
| 7 | Drawing Multiple Shapes on Images | Draws shapes (lines, rectangles, circles, polygons) on images using OpenCV drawing functions. | draw_shapes.py |
| 8 | Edge Detection | Applies edge detection algorithms (e.g., Canny, Sobel) to identify edges in images. | edge_detection.py |
| 9 | Hand Detection Using Webcam | Detects hands in real-time webcam feed using pre-trained models or color-based segmentation. | hand_detection_webcam.py |
| 10 | Hand Detection Using Contour in Images | Detects hands in static images using contour detection and shape analysis. | hand_detection_contour.py |
| 11 | Image Analysis Using Histogram | Analyzes image intensity distributions using histograms (grayscale or color channels). | image_histogram.py |
| 12 | Image Border Detection | Detects and highlights image borders using gradient-based techniques or contour analysis. | border_detection.py |
| 13 | Image Blending | Blends two images using weighted addition or alpha blending for seamless transitions. | image_blending.py |
| 14 | Object Detection Using Contour | Detects objects in images by identifying and analyzing contours based on shape or size. | object_detection_contour.py |
| 15 | Image Operations | Performs basic image operations (e.g., resizing, cropping, thresholding, morphological transforms). | image_operations.py |
| 16 | Image Pyramid | Creates image pyramids (Gaussian, Laplacian) for multi-scale image analysis. | image_pyramid.py |
| 17 | Inserting Shapes and Text on Videos | Adds shapes and text overlays to video frames in real-time or pre-recorded videos. | shapes_text_video.py |

## Repository Structure

```
computer-vision-labs/
├── data/
│   ├── sample_image.jpg              # Sample image for processing
│   ├── sample_video.mp4             # Sample video for frame extraction
├── notebooks/
│   ├── bitwise_operations.py
│   ├── extract_frames.py
│   ├── color_detection.py
│   ├── color_picker.py
│   ├── flip_operations.py
│   ├── contour_functions.py
│   ├── draw_shapes.py
│   ├── edge_detection.py
│   ├── hand_detection_webcam.py
│   ├── hand_detection_contour.py
│   ├── image_histogram.py
│   ├── border_detection.py
│   ├── image_blending.py
│   ├── object_detection_contour.py
│   ├── image_operations.py
│   ├── image_pyramid.py
│   ├── shapes_text_video.py
├── static/
│   ├── images/
│   │   ├── bitwise_output.png       # Output from bitwise operations
│   │   ├── frame_001.jpg            # Extracted video frame
│   │   ├── color_mask.png           # Color detection mask
│   │   ├── edges_canny.png          # Edge detection output
│   │   ├── hand_contour.png         # Hand detection output
│   │   ├── histogram_rgb.png        # RGB histogram
│   │   ├── blended_image.png        # Blended image output
│   │   ├── video_shapes_text.mp4    # Video with shapes/text
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── LICENSE                            # MIT License
```

## Related Coursework

This project builds on my prior coursework, particularly:

- **Lab 3: CNN Classification** (`deep-learning-labs/lab_manuals/CNN_Classification.pdf`): Covers CNN fundamentals, relevant to hand detection (Lab 9) if using pre-trained models.
- **Lab 4: CNN Patterns** (`deep-learning-labs/lab_manuals/CNN_Patterns.pdf`): Discusses image preprocessing (e.g., edge detection, histograms), applied in Labs 8, 11, 12.
- **Lab 8: Speech Signal Classification** (`speech-processing-labs/lab_reports/Lab8_Speech_Classification.pdf`): Feature extraction techniques, similar to contour and histogram analysis.

See the `deep-learning-labs` and `speech-processing-labs` repositories for details.

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/computer-vision-labs.git
   cd computer-vision-labs
   ```

2. **Install Dependencies**:

   Install Python libraries listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   Key libraries: `opencv-python`, `numpy`, `matplotlib`.

3. **Prepare Data**:

   - Place sample images (e.g., `sample_image.jpg`) and videos (e.g., `sample_video.mp4`) in the `data/` folder.
   - For hand detection (Lab 9), ensure a webcam is connected or use pre-recorded videos.

4. **Run Scripts**:

   Execute Python scripts in the `notebooks/` folder:

   ```bash
   python notebooks/<script_name>.py
   ```

   For Jupyter notebooks (if used):

   ```bash
   jupyter notebook notebooks/<notebook_name>.ipynb
   ```

5. **View Outputs**:

   Visualizations and processed outputs are saved in `static/images/` (e.g., `edges_canny.png`, `video_shapes_text.mp4`).

## Usage

- **Lab 1**: Apply bitwise operations (e.g., AND, OR) to mask regions of an image.
- **Lab 2**: Extract frames from `sample_video.mp4` and save as images in `static/images/`.
- **Lab 3**: Detect specific colors (e.g., red) in an image using HSV masking.
- **Lab 4**: Create an interactive color picker to display RGB/HSV values on mouse click.
- **Lab 5**: Flip images horizontally, vertically, or both using OpenCV’s `flip()` function.
- **Lab 6**: Detect and draw contours for object boundaries in images.
- **Lab 7**: Draw multiple shapes (e.g., circles, rectangles) on images.
- **Lab 8**: Apply Canny edge detection to highlight edges in grayscale images.
- **Lab 9**: Detect hands in real-time webcam feed using color segmentation or pre-trained models.
- **Lab 10**: Detect hands in static images using contour analysis and shape properties.
- **Lab 11**: Generate and plot histograms for RGB or grayscale image channels.
- **Lab 12**: Detect image borders using Sobel gradients or contour methods.
- **Lab 13**: Blend two images using weighted addition (e.g., `cv2.addWeighted`).
- **Lab 14**: Detect objects (e.g., cups, books) using contour area and shape analysis.
- **Lab 15**: Perform operations like resizing, cropping, and thresholding on images.
- **Lab 16**: Create Gaussian and Laplacian pyramids for multi-scale analysis.
- **Lab 17**: Overlay shapes and text on video frames, saving as a new video.

**Example** (Color Detection Using Mask):

```python
import cv2
import numpy as np

image = cv2.imread("data/sample_image.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)
result = cv2.bitwise_and(image, image, mask=mask)
cv2.imwrite("static/images/color_mask.png", result)
cv2.imshow("Red Detection", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Run in `notebooks/color_detection.py`.

## Future Improvements

- **Hand Detection (Lab 9)**: Integrate deep learning models (e.g., MediaPipe, YOLO) for robust hand detection, linking to `deep-learning-labs/lab_manuals/CNN_Classification.pdf`.
- **Web Interface**: Develop a Flask-based interface (similar to `sales-forecasting/app.py`) for interactive image/video processing.
- **Real-Time Optimization**: Optimize Labs 9 and 17 for real-time performance on low-resource devices.
- **Advanced Features**: Add HOG or SIFT for feature-based object detection in Lab 14.
- **Visualization**: Incorporate interactive plots (e.g., Plotly) for histograms (Lab 11).

## Notes

- **File Size**: Use Git LFS for large files (e.g., `git lfs track "*.jpg" "*.mp4" "*.png"`).
- **Hardware**: Labs 9 and 17 require a webcam for real-time processing; use pre-recorded videos as alternatives.
- **Dependencies**: Ensure OpenCV is installed with `pip install opencv-python`.

## License

This repository is licensed under the MIT License. See the LICENSE file for details.