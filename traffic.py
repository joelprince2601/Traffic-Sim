import cv2
import numpy as np
import os
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

def get_yolo_net(cfg_file, weights_file, names_file):
    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Weights file not found: {weights_file}")
    if not os.path.exists(names_file):
        raise FileNotFoundError(f"Names file not found: {names_file}")

    net = cv2.dnn.readNet(weights_file, cfg_file)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open(names_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, output_layers, classes

def detect_objects(net, output_layers, frame, input_size=(416, 416)):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id in [5, 1, 2]:  # Only detect persons, bicycles, and cars
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    return [boxes[i] for i in indices], [confidences[i] for i in indices], [class_ids[i] for i in indices]

def draw_labels(frame, trackers):
    for tracker in trackers:
        if not tracker.is_confirmed() or tracker.time_since_update > 1:
            continue

        bbox = tracker.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure the bounding box is within the frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue  # Skip invalid bounding boxes

        # Calculate center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Draw blue circle at the center
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

def draw_grid(frame):
    height, width = frame.shape[:2]
    step_x = width // 3
    step_y = height // 3

    for i in range(1, 3):
        x = i * step_x
        cv2.line(frame, (x, 0), (x, height), (255, 0, 0), 2)

    for i in range(1, 3):
        y = i * step_y
        cv2.line(frame, (0, y), (width, y), (255, 0, 0), 2)

class CarCounter:
    def __init__(self):
        self.cars_in_focus = set()

    def update(self, trackers, frame_shape):
        height, width = frame_shape[:2]
        step_y = height // 3
        
        current_cars = set()
        for tracker in trackers:
            if not tracker.is_confirmed() or tracker.time_since_update > 1:
                continue
            
            bbox = tracker.to_tlbr()
            center_y = (bbox[1] + bbox[3]) / 2
            
            if center_y > step_y:  # Check if the car is below the first row
                current_cars.add(tracker.track_id)
        
        # Add new cars to the focus area
        self.cars_in_focus.update(current_cars)
        
        # Remove cars that are no longer detected
        self.cars_in_focus = self.cars_in_focus.intersection(current_cars)
        
        return len(self.cars_in_focus)

class TrafficLightController:
    def __init__(self):
        self.states = ['Red', 'Green']
        self.current_state = 'Red'
        self.car_threshold = 3
        self.time_threshold = 2  # 2 seconds
        self.start_time = None

    def update(self, car_count, current_time):
        if self.current_state == 'Red' and car_count >= self.car_threshold:
            if self.start_time is None:
                self.start_time = current_time
            elif current_time - self.start_time >= self.time_threshold:
                self.current_state = 'Green'
                self.start_time = None
        elif self.current_state == 'Green' and car_count < self.car_threshold:
            if self.start_time is None:
                self.start_time = current_time
            elif current_time - self.start_time >= self.time_threshold:
                self.current_state = 'Red'
                self.start_time = None
        else:
            self.start_time = None
        
        return self.current_state

def main(video_path):
    # Load YOLO network
    cfg_file = "yolov4-tiny.cfg"
    weights_file = "yolov4-tiny.weights"
    names_file = "coco.names"

    try:
        net, output_layers, classes = get_yolo_net(cfg_file, weights_file, names_file)
    except FileNotFoundError as e:
        print(e)
        return

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Initialize DeepSort tracker
    deepsort = DeepSort(max_age=30, n_init=3)
    car_counter = CarCounter()
    traffic_light = TrafficLightController()

    # Get the original video dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the new dimensions while maintaining aspect ratio
    target_width = 640
    target_height = int(target_width * original_height / original_width)

    frame_count = 0
    start_time = time.time()

    # Get input for frame rate
    frame_rate = float(input("Enter the desired frame rate (fps): "))
    frame_delay = 1 / frame_rate

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()

        # Resize the frame to the target dimensions
        frame = cv2.resize(frame, (target_width, target_height))

        # Prepare input frame for object detection
        input_frame = cv2.resize(frame, (416, 416))
        boxes, confidences, class_ids = detect_objects(net, output_layers, input_frame)
        
        # Prepare detections for DeepSort
        detections = []
        for box, class_id, conf in zip(boxes, class_ids, confidences):
            x, y, w, h = box
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Ensure the bounding box is within the frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(target_width - 1, x2), min(target_height - 1, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue  # Skip invalid bounding boxes
            
            # Create a detection in the format expected by DeepSort
            detection = ([x1, y1, x2, y2], conf, class_id)
            detections.append(detection)

        try:
            # Update tracks using DeepSort
            trackers = deepsort.update_tracks(detections, frame=frame)
        except Exception as e:
            print(f"Error updating tracks: {e}")
            trackers = []

        # Draw labels for tracked objects
        draw_labels(frame, trackers)

        # Draw grid on the frame
        draw_grid(frame)

        # Count cars in the focus area
        car_count = car_counter.update(trackers, frame.shape)

        # Update traffic light state
        light_state = traffic_light.update(car_count, current_time)

        # Draw traffic light
        light_color = (0, 0, 255) if light_state == 'Red' else (0, 255, 0)
        cv2.circle(frame, (50, 50), 30, light_color, -1)
        cv2.putText(frame, light_state, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, light_color, 2)

        # Display car count
        cv2.putText(frame, f"Cars in focus area: {car_count}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('Car Detection & Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Control frame rate
        time.sleep(max(0, frame_delay - (time.time() - start_time - elapsed_time)))

    # Release video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

    print(f"Average FPS: {frame_count / elapsed_time:.2f}")

if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ").strip()
    main(video_path)