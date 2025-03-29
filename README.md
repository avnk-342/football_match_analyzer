# Video Object Detection and Tracking

This repository contains a pipeline for object detection, tracking, and team assignment in videos. It leverages YOLOv8 for object detection and includes utilities for bounding box manipulation, team assignment, and video processing.

## Project Structure
├── main.py # Main script to run the pipeline 
├── prediction.py # Script for making predictions 
├── development_and_analysis/ # Jupyter notebooks for analysis and development 
├── input_videos/ # Input video files 
├── models/ # Pre-trained YOLOv8 models 
├── output_videos/ # Processed output videos and images 
├── runs/ # YOLOv8 run outputs 
├── stubs/ # Stub files for testing 
├── team_assignment/ # Team assignment logic 
├── tracking/ # Object tracking logic 
├── training/ # YOLOv8 training scripts 
├── utils/ # Utility functions 


## Usage
1. Run the Pipeline
To process a video, use the main.py script:
    python3 main.py

2. Train YOLOv8
To train the YOLOv8 model, use the notebook in training/yolov8.ipynb.

3. Analyze Results
Use the Jupyter notebooks in development_and_analysis/ for further analysis.

## File Descriptions
main.py: Entry point for the pipeline.\n
prediction.py: Handles predictions using the trained model.
team_assignment/team_assigner.py: Implements team assignment logic.
tracking/tracker.py: Contains object tracking logic.
utils/bbox_utils.py: Utility functions for bounding box operations.
Input and Output
Input Videos: Place input videos in the input_videos/ directory.
Output Videos: Processed videos and images are saved in the output_videos/ directory.
Pre-trained Models
Pre-trained YOLOv8 models are stored in the models/ directory:

best.pt: Best-performing model.
last.pt: Most recent model.

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
1. Ultralytics YOLOv8 for the object detection framework.
2. Open-source libraries and tools used in this project.

## Contact
For questions or feedback, please open an issue or contact the repository owner.

