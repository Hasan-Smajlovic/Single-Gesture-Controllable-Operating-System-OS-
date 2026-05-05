import cv2 
import csv 
import os 
from datetime import datetime
from camera.camera import Camera
from gestures.hand_tracker import HandTracker

GESTURES = ["close","zoom in","zoom out","minimize", "maximize", "scroll up", "scroll down"] #make an empty array for my gestures

CSV_FILE = "data/gestures_dataset.csv" # define the path to the csv file

def setup_csv(): # function to set up my csv file
    if not os.path.exists("data"):
        print("Our directory is not existant, will need to create it.")
        os.makedirs("data") #this is how i am creating my data directory if it doesn't exist
        print("Directory created successfully.")
    if not os.path.isfile(CSV_FILE):
        print("The CSV file is not found, will create it then.")
        with open(CSV_FILE, mode = "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["gesture", "timestamp"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]) #write the header row with gesture, timestamp, and landmark coordinates
        print("CSV file created successfully.")

def extract_landmarks(hand_landmarks): #function to extract my hand_landmarks
    hand_landmarks_list = []
    for landmark in hand_landmarks:  # hand_landmarks is already a listq
        hand_landmarks_list.append(landmark.x) #append the x coordinate of the landmark to the list
        hand_landmarks_list.append(landmark.y) #append the y coordinate of the landmark to the list
        # hand_landmarks_list.append(handlandmark.z) #used for depth of camera, wont be using it now
    return hand_landmarks_list #return the list of landmarks

def save_gestures(gesture_name, hand_landmarks):#function to save my landmarks to the csv
    if not os.path.isfile(CSV_FILE):
        print("CSV file not found, cannot save gesture data.")
        return
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        timestamp = datetime.now().isoformat() #get the current timestamp in ISO format
        writer.writerow([gesture_name, timestamp] + hand_landmarks) #write a new row to the csv with the gesture name, timestamp, and landmark coordinates
    print(f"Gesture '{gesture_name}' saved successfully at {timestamp}.")


