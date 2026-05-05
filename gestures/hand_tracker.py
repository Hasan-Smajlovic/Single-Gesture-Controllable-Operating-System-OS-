"""
Hand Tracker Module - Detects and tracks 21 hand landmarks using MediaPipe
"""

# Import OpenCV for image processing and drawing
import cv2 as cv

# Import MediaPipe for hand detection AI model
import mediapipe as mp

# Import the MediaPipe Tasks Python API
from mediapipe.tasks import python as mp_python

# Import the vision module for hand landmark detection
from mediapipe.tasks.python import vision


class HandTracker: 
    """Class to detect hand landmarks from video frames"""
    
    def __init__(self, model_path="hand_landmarker.task"):#  Initialize the hand tracker with MediaPipe hand landmarker model

        # Create base options with the path to the AI model file
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        print("Loading hand landmark model...")
        
        # Configure the hand landmarker options
        options = vision.HandLandmarkerOptions(
            base_options=base_options,  # Use the base options we created
            running_mode=vision.RunningMode.VIDEO,  # Process video frames (not single images)
            num_hands=2,  # Detect up to 2 hands in the frame
            min_hand_detection_confidence=0.5,  # Minimum confidence (0-1) to detect a hand
            min_hand_presence_confidence=0.5,  # Minimum confidence hand is present
            min_tracking_confidence=0.5  # Minimum confidence for tracking across frames
        )
        print("Hand landmark model loaded successfully.")
        
        # Create the hand landmarker from the configured options
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        print("Initializing hand landmarker...")
        
        # Initialize timestamp for video frame tracking
        self.timestamp_ms = 0
        print("Hand landmarker initialized.")

        # Define which landmarks connect to each other (the hand skeleton)
        # Format: (point_from, point_to) - these are the lines drawn between landmarks
        self.HAND_CONNECTIONS = [
            # Thumb connections
            (0,1),(1,2),(2,3),(3,4),        
            # Index finger connections
            (0,5),(5,6),(6,7),(7,8),        
            # Middle finger connections
            (0,9),(9,10),(10,11),(11,12),   
            # Ring finger connections
            (0,13),(13,14),(14,15),(15,16), 
            # Pinky finger connections
            (0,17),(17,18),(18,19),(19,20), 
            # Palm connections (connecting fingers to palm)
            (5,9),(9,13),(13,17)            
        ]
        print("Hand connections defined.")

    def detect(self, frame): #Detect hand landmarks in a video frame

        # Increment timestamp by 33ms (roughly 30 FPS)
        self.timestamp_ms += 33
        
        # Convert frame from BGR color format to RGB (MediaPipe expects RGB)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  
        
        # Wrap the RGB frame in MediaPipe's Image format with SRGB color space
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  
                
        # Check if landmarker was initialized properly
        if self.landmarker is None:
            print("Error: Hand la\ndmarker not initialized.")
            return None
        
        # Run hand detection on the frame with the timestamp
        # This returns the 21 landmark points for each detected hand
        return self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
    
    def draw_landmarks(self, frame, results):  #Draw hand landmarks (points and connections) on the frame
        # Check if any hands were detected
        if not results.hand_landmarks:
            return frame
        
        # Get frame dimensions (height and width in pixels)
        h, w = frame.shape[:2]
        
        # Loop through each detected hand
        for hand in results.hand_landmarks: 
            # Draw lines (connections) between landmark points
            for a, b in self.HAND_CONNECTIONS:
                # Convert normalized coordinates (0-1) to pixel coordinates
                x1 = int(hand[a].x * w)  # X coordinate of first point
                y1 = int(hand[a].y * h)  # Y coordinate of first point
                x2 = int(hand[b].x * w)  # X coordinate of second point
                y2 = int(hand[b].y * h)  # Y coordinate of second point
                
                # Draw a green line from point (x1,y1) to (x2,y2) with thickness 2
                cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw circles at each landmark point
            for lm in hand:  # lm = landmark (one of the 21 points)
                # Convert normalized X coordinate to pixel coordinate
                cx = int(lm.x * w)
                # Convert normalized Y coordinate to pixel coordinate
                cy = int(lm.y * h)
                # Draw a red filled circle at (cx, cy) with radius 5
                cv.circle(frame, (cx, cy), 5, (0, 0, 255), cv.FILLED)  
        
        # Return the frame with all landmarks drawn on it
        return frame  

    def release(self):
        # Close the landmarker and free up memory
        self.landmarker.close()
        print("Hand landmarker released.")