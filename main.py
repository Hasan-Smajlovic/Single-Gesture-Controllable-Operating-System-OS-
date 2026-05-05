import cv2 as cv

from camera.camera import Camera
from gestures.hand_tracker import HandTracker
from training.collect_data import GESTURES, setup_csv, extract_landmarks, save_gestures

def main():
    setup_csv()
    camera = Camera()
    hand_tracker = HandTracker()
    
    print("\n" + "="*50)
    print("GESTURE DATA COLLECTION")
    print("="*50)
    print(f"Gestures to collect: {GESTURES}")
    print("\nControls:")
    print("  SPACE → Record current frame")
    print("  N     → Move to next gesture")
    print("  Q     → Quit")
    print("="*50 + "\n")

    current_gesture_index = 0
    frames_recorded = 0

    while True:
        ret, frame = camera.read()
        if not ret: 
            print("Failed to capture frame. Camera may be disconnected.")
            break
        
        results = hand_tracker.detect(frame)
        frame = hand_tracker.draw_landmarks(frame, results)
        
        # Display gesture name on screen
        cv.putText(frame, f"Gesture: {GESTURES[current_gesture_index]} | Recorded: {frames_recorded}", 
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display landmark count
        landmark_count = len(results.hand_landmarks[0]) if results.hand_landmarks else 0
        cv.putText(frame, f"Landmarks Detected: {landmark_count}", 
                   (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) 
        
        cv.putText(frame, f"Press SPACE to record, N for next gesture, Q to quit", 
                   (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
        
        cv.imshow("Gesture Data Collection", frame)

        key = cv.waitKey(1) & 0xFF
        
        if key == ord(' '):  # SPACE to record
            hand_landmarks = results.hand_landmarks if results else []
            if hand_landmarks:
                landmarks_list = extract_landmarks(hand_landmarks[0])  # Get first hand
                save_gestures(GESTURES[current_gesture_index], landmarks_list)
                frames_recorded += 1
                print(f"Recorded frame {frames_recorded}")
            else:
                print("No hand detected")
        
        elif key == ord('n'):  # N to move to next gesture
            current_gesture_index = (current_gesture_index + 1) % len(GESTURES)
            frames_recorded = 0
            print(f"\n→ Switched to: {GESTURES[current_gesture_index]}\n")

        elif key == ord('q'):  # Q to quit
            print("\nExiting...")
            break

    camera.release()
    hand_tracker.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()