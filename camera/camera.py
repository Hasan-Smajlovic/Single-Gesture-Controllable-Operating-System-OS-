import cv2 as cv


class Camera:  
    def __init__(self): #using init as a constructor to initialize the camera (0 is default webcam)
        self.camera = cv.VideoCapture(0) #opening my camera using OpenCV's VideoCapture class
        if not self.camera.isOpened():
            raise RuntimeError("Could not open webcam")
    
    def read(self):
        return self.camera.read() #get a frame from the camera, and send it back to the caller (returns a boolean and the frame itself) 
    ## returns ret, frame
    
    def release(self):
        self.camera.release() #release the camera resource when done, closes the webcam (saves the resource for other applications)