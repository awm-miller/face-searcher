# Face Searcher

A powerful face recognition tool that processes videos to identify and locate known faces. This project uses state-of-the-art face recognition technology to detect and match faces from a library of known individuals in video content.

## Features

- **Video Face Recognition**: Process videos to identify known faces with timestamp tracking
- **Face Library Management**: Maintain a library of known faces for recognition
- **Multiple Detection Models**: Support for both CPU (HOG) and GPU (CNN) detection models
- **Face Image Augmentation**: Automatic face image augmentation to improve recognition accuracy
- **Performance Optimization**: 
  - Caching system for face encodings
  - Configurable frame sampling rate
  - GPU acceleration support (when available)
- **Detailed Results**: Generate comprehensive results including timestamps and confidence scores

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for CNN model)
- See `requirements.txt` for Python package dependencies

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-searcher.git
   cd face-searcher
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On Unix or MacOS
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your faces library:
   - Create a directory named `faces_library`
   - Add subdirectories for each person, named after them
   - Place face images in respective subdirectories

2. Run face recognition on a video:
   ```bash
   python video_face_recognition.py --input path/to/video.mp4 --output path/to/output.mp4
   ```

### Command Line Arguments

- `--input`: Path to input video file
- `--output`: Path for output video file
- `--model`: Face detection model ('hog' or 'cnn', default: 'hog')
- `--frame-sample-rate`: Process every Nth frame (default: 2)
- `--faces-dir`: Path to faces library directory (default: 'faces_library')

## Project Structure

- `video_face_recognition.py`: Main script for video processing and face recognition
- `augment_faces.py`: Utilities for face image augmentation
- `faces_library/`: Directory containing known face images
- `results/`: Directory for storing recognition results
- `tests/`: Unit tests

## Performance Tips

1. Use the 'hog' model for CPU-only systems
2. Use the 'cnn' model if you have a CUDA-compatible GPU
3. Adjust frame sample rate based on your needs
4. Ensure face images in the library are clear and well-lit

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.