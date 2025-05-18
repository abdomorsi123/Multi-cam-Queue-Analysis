import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import random
import time
from scipy.spatial.distance import cosine
from collections import defaultdict
import config


def process_videos(video_path1=config.VIDEO_PATH1, 
                  video_path2=config.VIDEO_PATH2, 
                  output_path=config.OUTPUT_PATH):
    
    # Initialize YOLO model
    model = YOLO(config.YOLO_MODEL)
    model.conf = config.CONFIDENCE_THRESHOLD
    
    # Open video captures
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Cannot open video files")
        return
    
    # Get video properties
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    # Setup output video
    total_width = width1 + width2
    max_height = max(height1, height2)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, max_height))
    
    # Set maximum display dimensions
    max_display_width = config.MAX_DISPLAY_WIDTH
    max_display_height = config.MAX_DISPLAY_HEIGHT
    
    # Calculate display scale
    scale = min(max_display_width/total_width, max_display_height/max_height)
    display_width = int(total_width * scale)
    display_height = int(max_height * scale)
    
    # Initialize DeepSORT trackers for each camera (these will assign temporary IDs)
    tracker1 = DeepSort(max_age=config.MAX_AGE, n_init=config.N_INIT, nn_budget=config.NN_BUDGET)
    tracker2 = DeepSort(max_age=config.MAX_AGE, n_init=config.N_INIT, nn_budget=config.NN_BUDGET)

    # Initialize feature extractor (ResNet-50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    feature_extractor.fc = torch.nn.Identity()  # Remove the classification head
    feature_extractor.eval()  # Set to evaluation mode

    # Define image transforms for feature extraction
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dictionary to store global IDs and their features
    global_id_counter = 0
    global_ids = {}  # Maps {camera_id: {local_id: global_id}}
    id_features = {}  # Maps {global_id: list_of_feature_vectors}
    id_last_seen = {}  # Maps {global_id: timestamp}
    id_camera_history = defaultdict(set)  # Maps {global_id: set_of_cameras}
    
    # Dictionary to store colors for each global ID
    id_colors = {}

    # Dictionary to store the first and last frame for each global ID
    id_times = {}

    # Dictionary to store the estimated waiting time for each global ID
    id_estimates = {}

    # Variables to track historical data for improving estimates
    historical_differences = []
    initial_estimated_time = config.INITIAL_ESTIMATED_TIME

    # Extract features for a person detection
    def extract_features(frame, box):
        x1, y1, w, h = map(int, box)  # Convert coordinates to integers
        x2, y2 = x1 + w, y1 + h
        
        # Ensure coordinates are within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1 or w < 20 or h < 40:
            return None  # Invalid box or too small
            
        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            return None
        
        # Skip if aspect ratio is unusual (likely not a person)
        aspect_ratio = h / w
        if aspect_ratio < 1.0 or aspect_ratio > 4.0:
            return None
            
        # Convert to RGB for PyTorch models
        person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        # Transform image for the model
        input_tensor = transform(person_img_rgb).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = feature_extractor(input_tensor).cpu().numpy().flatten()
            
        # Normalize feature vector
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features

    # Function to get global ID based on features
    def get_global_id(camera_id, local_id, features, bbox):
        nonlocal global_id_counter
        current_time = time.time()
        
        # Initialize dictionaries if needed
        if camera_id not in global_ids:
            global_ids[camera_id] = {}
            
        # If this local ID already has a global ID assigned in this camera
        if local_id in global_ids[camera_id]:
            global_id = global_ids[camera_id][local_id]
            
            # Update the features by adding to the list (keep history of appearances)
            if global_id in id_features:
                # Keep a limited history of features (max 5)
                if len(id_features[global_id]) >= 5:
                    id_features[global_id].pop(0)
                id_features[global_id].append(features)
            else:
                id_features[global_id] = [features]
                
            # Update last seen time
            id_last_seen[global_id] = current_time
            
            # Update camera history
            id_camera_history[global_id].add(camera_id)
            
            return global_id
        
        # Clean up expired IDs (not seen in the last 5 seconds)
        expired_ids = []
        for gid, last_seen in id_last_seen.items():
            if current_time - last_seen > 5.0:  # 5 second expiration
                expired_ids.append(gid)
                
        for gid in expired_ids:
            if gid in id_features:
                del id_features[gid]
            id_last_seen.pop(gid, None)
            
        # Don't try to match if this is from a camera where we've seen this ID before
        # to prevent oscillating IDs
        recently_seen_ids = []
        for gid in id_features:
            if camera_id in id_camera_history[gid]:
                recently_seen_ids.append(gid)
            
        # Try to match with existing global IDs from all cameras
        best_match_id = None
        best_match_score = config.FEATURE_MATCH_THRESHOLD
        
        # Use the average similarity across all stored feature vectors
        for gid, feat_list in id_features.items():
            # Skip if this ID has been seen in this camera recently (to prevent oscillation)
            if gid in recently_seen_ids:
                continue
                
            # Calculate average similarity across all stored features
            similarities = []
            for feat in feat_list:
                similarity = 1 - cosine(features, feat)  # Higher is better
                similarities.append(similarity)
                
            # Use the average similarity
            avg_similarity = sum(similarities) / len(similarities)
            
            # Track the best match
            if avg_similarity > best_match_score:
                best_match_score = avg_similarity
                best_match_id = gid
                
        # If we found a match, use that global ID
        if best_match_id is not None:
            global_ids[camera_id][local_id] = best_match_id
            
            # Add this feature to the feature history
            if len(id_features[best_match_id]) >= 5:
                id_features[best_match_id].pop(0)
            id_features[best_match_id].append(features)
            
            # Update last seen time
            id_last_seen[best_match_id] = current_time
            
            # Update camera history
            id_camera_history[best_match_id].add(camera_id)
            
            return best_match_id
            
        # If no match, create a new global ID
        global_id_counter += 1
        new_global_id = global_id_counter
        global_ids[camera_id][local_id] = new_global_id
        id_features[new_global_id] = [features]  # Start with a list containing one feature
        id_last_seen[new_global_id] = current_time
        id_camera_history[new_global_id].add(camera_id)
        
        return new_global_id

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Run detection on both frames
        with torch.no_grad():
            results1 = model(frame1, verbose=False)[0]
            results2 = model(frame2, verbose=False)[0]
        
        # Prepare detections for DeepSORT
        detections1 = []
        detections2 = []
        for result, detections in [(results1, detections1), (results2, detections2)]:
            boxes = result.boxes
            for box in boxes:
                if box.cls[0] == 0 and box.conf[0] >= model.conf:  # Only process persons
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # YOLO format: [x1, y1, x2, y2]
                    w, h = x2 - x1, y2 - y1  # Convert to [x, y, w, h]
                    conf = float(box.conf[0])
                    detections.append(([x1, y1, w, h], conf))  # DeepSORT format: [x, y, w, h], confidence
        
        # Update DeepSORT trackers
        tracked_objects1 = tracker1.update_tracks(detections1, frame=frame1)
        tracked_objects2 = tracker2.update_tracks(detections2, frame=frame2)
        
        # Track total number of people in the queue (using global IDs)
        total_people = set()

        # Process tracked objects
        current_frame = int(cap1.get(cv2.CAP_PROP_POS_FRAMES))  # Get the current frame number
        
        # Process camera 1
        for track in tracked_objects1:
            if not track.is_confirmed():  # Skip unconfirmed tracks
                continue
                
            local_id = track.track_id
            ltwh = track.to_ltwh()  # [x, y, w, h]
            
            # Extract features for this detection
            features = extract_features(frame1, ltwh)
            if features is None:
                continue
                
            # Get or assign global ID
            global_id = get_global_id(camera_id=1, local_id=local_id, features=features, bbox=ltwh)
            
            # Add to the set of people in the queue
            total_people.add(global_id)
            
            # Assign a unique color for each global ID
            if global_id not in id_colors:
                id_colors[global_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
            # Update the first and last frame for the global ID
            if global_id not in id_times:
                id_times[global_id] = {"first_frame": current_frame, "last_frame": current_frame}
                id_estimates[global_id] = initial_estimated_time  # Assign initial estimated time
            else:
                id_times[global_id]["last_frame"] = current_frame
                
            # Calculate the actual waiting time in seconds
            first_frame = id_times[global_id]["first_frame"]
            last_frame = id_times[global_id]["last_frame"]
            actual_time = (last_frame - first_frame) / fps  # Convert frames to seconds
            
            # Decrease the estimated waiting time
            id_estimates[global_id] = max(0, id_estimates[global_id] - (1 / fps))  # Decrease by 1 second per second
            
            # Draw bounding box and ID with estimated waiting time on the frame
            color = id_colors[global_id]
            x, y, w, h = map(int, ltwh)
            x2, y2 = x + w, y + h
            
            cv2.rectangle(frame1, (x, y), (x2, y2), color, 2)
            cv2.putText(frame1, f'GID {global_id} | {id_estimates[global_id]:.1f}s', (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Process camera 2
        for track in tracked_objects2:
            if not track.is_confirmed():  # Skip unconfirmed tracks
                continue
                
            local_id = track.track_id
            ltwh = track.to_ltwh()  # [x, y, w, h]
            
            # Extract features for this detection
            features = extract_features(frame2, ltwh)
            if features is None:
                continue
                
            # Get or assign global ID
            global_id = get_global_id(camera_id=2, local_id=local_id, features=features, bbox=ltwh)
            
            # Add to the set of people in the queue
            total_people.add(global_id)
            
            # Assign a unique color for each global ID
            if global_id not in id_colors:
                id_colors[global_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
            # Update the first and last frame for the global ID
            if global_id not in id_times:
                id_times[global_id] = {"first_frame": current_frame, "last_frame": current_frame}
                id_estimates[global_id] = initial_estimated_time  # Assign initial estimated time
            else:
                id_times[global_id]["last_frame"] = current_frame
                
            # Calculate the actual waiting time in seconds
            first_frame = id_times[global_id]["first_frame"]
            last_frame = id_times[global_id]["last_frame"]
            actual_time = (last_frame - first_frame) / fps  # Convert frames to seconds
            
            # Decrease the estimated waiting time
            id_estimates[global_id] = max(0, id_estimates[global_id] - (1 / fps))  # Decrease by 1 second per second
            
            # Draw bounding box and ID with estimated waiting time on the frame
            color = id_colors[global_id]
            x, y, w, h = map(int, ltwh)
            x2, y2 = x + w, y + h
            
            cv2.rectangle(frame2, (x, y), (x2, y2), color, 2)
            cv2.putText(frame2, f'GID {global_id} | {id_estimates[global_id]:.1f}s', (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Get all visible IDs from both cameras
        visible_ids = set()
        for track in tracked_objects1:
            if track.is_confirmed() and track.track_id in global_ids.get(1, {}):
                visible_ids.add(global_ids[1][track.track_id])
                
        for track in tracked_objects2:
            if track.is_confirmed() and track.track_id in global_ids.get(2, {}):
                visible_ids.add(global_ids[2][track.track_id])
                
        # Remove global IDs that are no longer visible in either camera
        for global_id in list(id_times.keys()):
            if global_id not in visible_ids:
                # Calculate the final waiting time for the ID
                final_wait_time = (id_times[global_id]["last_frame"] - id_times[global_id]["first_frame"]) / fps
                print(f"Global ID {global_id} waited for {final_wait_time:.1f} seconds.")

                # Update historical differences for improving estimates
                if global_id in id_estimates:
                    difference = final_wait_time - initial_estimated_time
                    historical_differences.append(difference)
                    if len(historical_differences) > config.MAX_HISTORICAL_DIFFERENCES:  
                        historical_differences.pop(0)
                    # Update the initial estimated time based on historical data
                    if historical_differences:
                        initial_estimated_time += sum(historical_differences) / len(historical_differences)
                        initial_estimated_time = max(config.MIN_ESTIMATED_TIME, min(initial_estimated_time, config.MAX_ESTIMATED_TIME))  # Clamp between 5 and 60 seconds

                # Remove the ID from tracking and feature storage
                del id_times[global_id]
                del id_estimates[global_id]
                if global_id in id_features:
                    del id_features[global_id]
                if global_id in id_last_seen:
                    del id_last_seen[global_id]
                if global_id in id_camera_history:
                    del id_camera_history[global_id]
                
                # Remove from global_ids mapping (both camera 1 and 2)
                for camera_id in global_ids:
                    for local_id, g_id in list(global_ids[camera_id].items()):
                        if g_id == global_id:
                            del global_ids[camera_id][local_id]

        # Combine frames
        combined_frame = np.hstack((frame1, frame2))
        
        # Display the total number of people in the queue
        num_people = len(total_people)
        cv2.putText(combined_frame, f'Queue: {num_people}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
        # Display the current estimation algorithm performance
        if historical_differences:
            avg_diff = sum(historical_differences) / len(historical_differences)
            # cv2.putText(combined_frame, f'Avg Diff: {avg_diff:.2f}s', (10, 60), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Resize for display only
        display_frame = cv2.resize(combined_frame, (display_width, display_height))
        
        # Show resized frame
        cv2.imshow('Person Detection', display_frame)
        
        # Write original resolution to video
        out.write(combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_videos()