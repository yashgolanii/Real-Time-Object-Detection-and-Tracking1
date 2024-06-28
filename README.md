
# Object Tracking and Location Reporting using YOLO and OpenCV

## Project Overview
This project leverages the YOLOv8 object detection model and a custom tracking algorithm to process video feeds or sequences of images. The detected objects are tracked across frames, and their GPS locations are reported to a ground station. The project demonstrates the integration of computer vision and real-time location reporting for applications such as surveillance, autonomous vehicles, and more.

### Key Features
- **Object Detection**: Uses YOLOv8 to detect objects in video frames or images.
- **Object Tracking**: Tracks detected objects across multiple frames.
- **Location Reporting**: Reports the GPS location of tracked objects to a ground station.
- **Real-time Processing**: Processes video feed in real-time and displays results with bounding boxes and object IDs.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
  - [Loading the YOLO Model](#loading-the-yolo-model)
  - [Processing Frames](#processing-frames)
  - [Video and Image Processing](#video-and-image-processing)
- [License](#license)
- [Contact](#contact)

## Installation
1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/username/YOLO_Object_Tracking.git
   cd YOLO_Object_Tracking
   \`\`\`

2. Install the required dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. Download the YOLOv8 model weights and place them in the project directory:
   \`\`\`bash
   # Download from the official YOLO website or use the command
   wget https://path.to/yolov8s.pt
   \`\`\`

## Usage
1. Ensure you have your video file or image folder in the specified path.

2. Run the script to process video feed or images.

   For video feed:
   \`\`\`python
   process_video('path/to/your/video.mp4')
   \`\`\`

   For image sequences:
   \`\`\`python
   process_images('/path/to/your/image_folder')
   \`\`\`

## Code Explanation

### Loading the YOLO Model
The YOLOv8 model is loaded with the following line:
\`\`\`python
model = YOLO('yolov8s.pt')
\`\`\`
The model weights should be placed in the project directory.

### Processing Frames
The \`process_frame\` function is responsible for resizing frames, running the YOLO model for detection, and tracking the detected objects.

\`\`\`python
def process_frame(frame):
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        if d >= len(class_list):
            print(f"Warning: Detected class ID {d} is out of range for class_list.")
            continue  # Skip this detection if class ID is out of range
        c = class_list[d]
        list.append([x1, y1, x2, y2, d])  # Include the class id in the list
    bbox_id = tracker.update([bbox[:4] for bbox in list])  # Pass only bbox coordinates
    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)
        cv2.circle(frame, (cx, cy), 3, (255, 0, 255), -1)
        cv2.putText(frame, str(obj_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        if obj_id not in reported_ids:
            send_location(obj_id)  # Function to send GPS location
            reported_ids.append(obj_id)
        else:
            print(obj_id, "- already sent")
    return frame
\`\`\`

### Video and Image Processing
- **Video Feed**:
  \`\`\`python
  def process_video(video_path):
      cap = cv2.VideoCapture(video_path)
      while True:
          ret, frame = cap.read()
          if not ret:
              break
          frame = process_frame(frame)
          cv2.imshow("RGB", frame)
          if cv2.waitKey(1) & 0xFF == 27:
              break
      cap.release()
      cv2.destroyAllWindows()
  \`\`\`

- **Image Sequence**:
  \`\`\`python
  def process_images(image_folder):
      images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
      images.sort()
      for image in images:
          frame = cv2.imread(os.path.join(image_folder, image))
          frame = process_frame(frame)
          cv2.imshow("RGB", frame)
          if cv2.waitKey(1) & 0xFF == 27:
              break
      cv2.destroyAllWindows()
  \`\`\`

### Tracker Instance
A \`Tracker\` instance is used to keep track of detected objects across frames:
\`\`\`python
tracker = Tracker()
reported_ids = []
\`\`\`

### Mouse Callback Function
The \`RGB\` function is used to capture mouse movements over the display window:
\`\`\`python
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
\`\`\`

### Sending Location
The \`send_location\` function implements the logic to send GPS locations to the ground station:
\`\`\`python
def send_location(obj_id):
    print("location sent - ", obj_id)
    # Implement the logic to send GPS location to the ground station
    pass
\`\`\`

## License
This project is licensed under the MIT License.

## Contact
For any questions or contributions, please contact [yashgolani2004gmail.com].
