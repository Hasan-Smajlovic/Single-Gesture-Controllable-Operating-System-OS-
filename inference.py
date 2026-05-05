from pathlib import Path
import time

import cv2 as cv
import torch

from camera.camera import Camera
from commands import execute_command
from gestures.hand_tracker import HandTracker
from model.gesture_model import GestureModel


MODEL_PATH = Path("model/gesture_model.pth")
CONFIDENCE_THRESHOLD = 0.80
REQUIRED_STREAK = 5
ACTION_COOLDOWN_SECONDS = 1.2
DEFAULT_DRY_RUN = True


def extract_landmarks(hand_landmarks):
    landmark_values = []
    for landmark in hand_landmarks:
        landmark_values.append(landmark.x)
        landmark_values.append(landmark.y)
    return landmark_values


def load_gesture_model(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Train the model first.")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        class_names = checkpoint.get("class_names")
        if not class_names:
            raise ValueError("Checkpoint is missing class_names. Re-train to regenerate checkpoint.")

        input_shape = checkpoint.get("input_shape", 42)
        hidden_units = checkpoint.get("hidden_units", 128)
        output_shape = checkpoint.get("output_shape", len(class_names))
        state_dict = checkpoint["model_state_dict"]
    else:
        # Backward compatibility with old state_dict-only files.
        class_names = ["close", "maximize", "minimize", "zoom in", "zoom out"]
        input_shape = 42
        hidden_units = 128
        output_shape = len(class_names)
        state_dict = checkpoint

    model = GestureModel(
        input_shape=input_shape,
        hidden_units=hidden_units,
        output_shape=output_shape,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, class_names


def predict_gesture(model, landmarks, labels, device):
    input_tensor = torch.tensor([landmarks], dtype=torch.float32, device=device)

    with torch.inference_mode():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, class_index = torch.max(probabilities, dim=1)

    gesture = labels[class_index.item()]
    return gesture, confidence.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, labels = load_gesture_model(device)

    camera = Camera()
    hand_tracker = HandTracker()

    print("\n" + "=" * 50)
    print("GESTURE INFERENCE")
    print("=" * 50)
    print("Controls:")
    print("  Q -> Quit")
    print("  E -> Enable/Disable command execution")
    print("  R -> Toggle dry-run mode")
    print("=" * 50 + "\n")

    commands_enabled = True
    dry_run = DEFAULT_DRY_RUN
    last_trigger_time = 0.0
    last_gesture = None
    gesture_streak = 0

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture frame. Camera may be disconnected.")
                break

            results = hand_tracker.detect(frame)
            frame = hand_tracker.draw_landmarks(frame, results)

            gesture_text = "No hand detected"
            confidence_text = "-"
            color = (0, 0, 255)
            status_text = f"Commands: {'ON' if commands_enabled else 'OFF'} | Dry-run: {'ON' if dry_run else 'OFF'}"

            if results and results.hand_landmarks:
                landmarks = extract_landmarks(results.hand_landmarks[0])
                gesture, confidence = predict_gesture(model, landmarks, labels, device)

                if confidence >= CONFIDENCE_THRESHOLD:
                    gesture_text = gesture
                    confidence_text = f"{confidence:.2f}"
                    color = (0, 255, 0)

                    if gesture == last_gesture:
                        gesture_streak += 1
                    else:
                        last_gesture = gesture
                        gesture_streak = 1

                    now = time.time()
                    can_fire = (now - last_trigger_time) >= ACTION_COOLDOWN_SECONDS
                    is_stable = gesture_streak >= REQUIRED_STREAK

                    if commands_enabled and is_stable and can_fire:
                        success = execute_command(gesture, dry_run=dry_run)
                        if success:
                            last_trigger_time = now
                            status_text = f"Triggered: {gesture}"
                else:
                    gesture_text = "Uncertain"
                    confidence_text = f"{confidence:.2f}"
                    color = (0, 165, 255)
                    last_gesture = None
                    gesture_streak = 0
            else:
                last_gesture = None
                gesture_streak = 0

            cv.putText(
                frame,
                f"Predicted Gesture: {gesture_text}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            cv.putText(
                frame,
                f"Confidence: {confidence_text}",
                (10, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            cv.putText(
                frame,
                f"Stable frames: {gesture_streak}/{REQUIRED_STREAK}",
                (10, 90),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            cv.putText(
                frame,
                status_text,
                (10, 120),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv.imshow("Gesture Recognition", frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("e"):
                commands_enabled = not commands_enabled
                print(f"Command execution {'enabled' if commands_enabled else 'disabled'}.")
            if key == ord("r"):
                dry_run = not dry_run
                print(f"Dry-run {'enabled' if dry_run else 'disabled'}.")
    finally:
        camera.release()
        hand_tracker.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
