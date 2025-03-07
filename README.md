# Hand Gesture Even-Odd Counter

A computer vision application that detects hand gestures and counts extended fingers in real-time, indicating whether the count is even or odd. Built with OpenCV and MediaPipe.

![Hand Gesture Counter Demo](https://github.com/ajena555/Hand-gestured-eveodd/readme/screenshot.png)

## Features

- Real-time hand detection and finger counting
- Visual feedback showing even/odd status
- Support for both left and right hands
- Stability tracking for reliable gesture recognition
- Performance metrics display (FPS)
- Fullscreen interactive display

## Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- Git (for cloning the repository)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ajena555/Hand-gestured-eveodd.git
cd Hand-gestured-eveodd
```

### 2. Create and activate a virtual environment

#### For Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### For macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python modifiedevenodd.py
```

Press 'q' to exit the application.

## How It Works

1. The application uses your webcam to capture video frames.
2. MediaPipe's hand detection model identifies hand landmarks in the frame.
3. The algorithm analyzes finger positions to determine which fingers are extended.
4. The total count of extended fingers is calculated and displayed.
5. The application indicates whether the count is even or odd.
6. A stability measure shows how consistent the detection is over time.

## Usage Tips

- Ensure good lighting conditions for optimal hand detection.
- Position your hand clearly in the camera's view, palm facing the camera.
- Keep your hand movements slow and deliberate for better stability.
- For thumb detection, extend it clearly to the side.
- The application works best when your hand is approximately 12-24 inches from the camera.

## Requirements.txt

Create a file named `requirements.txt` with the following contents:

```
opencv-python>=4.6.0
mediapipe>=0.8.10
numpy>=1.20.0
```

## Troubleshooting

### Common Issues

1. **Poor detection accuracy:**
   - Improve lighting conditions
   - Position hand more clearly in the camera view
   - Ensure nothing else in the frame resembles a hand

2. **Low FPS:**
   - Close other applications using the camera or CPU
   - Reduce camera resolution in the code if needed

3. **MediaPipe installation errors:**
   - Ensure you're using a compatible Python version (3.8-3.10 recommended)
   - Try installing MediaPipe with `pip install mediapipe --no-deps`
   - For Apple Silicon Macs, use Rosetta 2 terminal to install dependencies

## Customization

You can modify the following parameters in the code:

- `min_detection_confidence`: Increase for better accuracy, decrease for better performance
- `min_tracking_confidence`: Adjust tracking stability vs. performance
- `gesture_stable_threshold`: Change how many consistent frames are required for stability
- Display colors and layout can be adjusted in the code

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the hand detection model
- [OpenCV](https://opencv.org/) for computer vision functionality