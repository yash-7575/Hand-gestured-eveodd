import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Hands with improved parameters
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Support for detecting two hands
    min_detection_confidence=0.6,  # Increased from default 0.5
    min_tracking_confidence=0.6    # Increased from default 0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to count fingers with improved logic
def count_fingers(landmarks, hand_type):
    # Define finger tip indices
    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    
    # Define indices for the middle knuckles (PIP joints)
    pip_joints = [
        mp_hands.HandLandmark.THUMB_IP,  # Interphalangeal joint for thumb
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]
    
    # Count extended fingers
    count = 0
    
    # Check if thumb is extended (special case)
    # For left hand, thumb is extended if its x-coordinate is less than that of the thumb IP joint
    # For right hand, thumb is extended if its x-coordinate is greater than that of the thumb IP joint
    if hand_type == "Right" and landmarks[finger_tips[0]].x > landmarks[pip_joints[0]].x:
        count += 1
    elif hand_type == "Left" and landmarks[finger_tips[0]].x < landmarks[pip_joints[0]].x:
        count += 1
    
    # Check other fingers by comparing y-coordinates (finger is extended if tip is above PIP joint)
    for i in range(1, 5):  # Index, middle, ring, pinky
        if landmarks[finger_tips[i]].y < landmarks[pip_joints[i]].y:
            count += 1
    
    return count

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create window
window_name = "Enhanced Finger Counter"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Variables for FPS calculation
prev_time = 0
curr_time = 0

# Variables for gesture history
gesture_history = []
gesture_stable_threshold = 10  # Number of frames for stable detection
current_stable_gesture = None
stable_count = 0

# Color theme
bg_color = (0, 0, 0)
text_color = (0, 255, 0)
highlight_color = (0, 255, 255)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Flip the frame horizontally for a more intuitive mirror view
    frame = cv2.flip(frame, 1)
    
    # Get frame dimensions
    h, w, _ = frame.shape
    
    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    
    # Convert the image to RGB (MediaPipe requires RGB input)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable
    image_rgb.flags.writeable = False
    
    # Process the image with MediaPipe Hands
    results = hands.process(image_rgb)
    
    # Set the image as writeable again for drawing
    image_rgb.flags.writeable = True
    
    # Create a correctly sized display frame
    display_width = w + 300  # Width of frame plus panel
    display_frame = np.zeros((h, display_width, 3), dtype=np.uint8)
    
    # Copy the original frame to the left side of the display
    video_region = frame.copy()
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Determine hand type (left or right)
            hand_type = "Unknown"
            if results.multi_handedness:
                hand_type = results.multi_handedness[idx].classification[0].label
            
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                video_region,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    # Copy the video region to display frame (safe copy)
    display_frame[0:h, 0:w] = video_region
    
    # Process hand landmarks if detected
    finger_count = 0
    hand_type = "Unknown"
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Determine hand type (left or right)
            if results.multi_handedness:
                hand_type = results.multi_handedness[idx].classification[0].label
            
            # Count fingers
            finger_count += count_fingers(hand_landmarks.landmark, hand_type)
    
    # Update gesture history
    gesture_history.append(finger_count)
    if len(gesture_history) > gesture_stable_threshold:
        gesture_history.pop(0)
    
    # Check if gesture is stable
    if all(g == gesture_history[0] for g in gesture_history) and len(gesture_history) == gesture_stable_threshold:
        if current_stable_gesture != gesture_history[0]:
            current_stable_gesture = gesture_history[0]
            stable_count = 0
        else:
            stable_count += 1
    
    # Add information panel
    info_panel_x = w + 10
    cv2.rectangle(display_frame, (info_panel_x, 0), (display_width, h), (30, 30, 30), -1)
    
    # Display finger count
    result_text = "Even" if finger_count % 2 == 0 else "Odd"
    cv2.putText(display_frame, f"Fingers: {finger_count}", (info_panel_x + 20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(display_frame, f"({result_text})", (info_panel_x + 20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    
    # Display hand type
    cv2.putText(display_frame, f"Hand: {hand_type}", (info_panel_x + 20, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    
    # Display FPS
    cv2.putText(display_frame, f"FPS: {int(fps)}", (info_panel_x + 20, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    
    # Display stability information
    stability = min(100, (stable_count / 30) * 100)  # Max 100%
    cv2.putText(display_frame, f"Stability: {int(stability)}%", (info_panel_x + 20, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    
    # Draw a colored indicator for even/odd
    indicator_color = (0, 255, 0) if finger_count % 2 == 0 else (0, 0, 255)  # Green for even, red for odd
    cv2.circle(display_frame, (info_panel_x + 150, 350), 40, indicator_color, -1)
    cv2.putText(display_frame, result_text, (info_panel_x + 120, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display instructions
    cv2.putText(display_frame, "Press 'q' to quit", (info_panel_x + 20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, highlight_color, 1)
    
    # Display the frame
    cv2.imshow(window_name, display_frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
