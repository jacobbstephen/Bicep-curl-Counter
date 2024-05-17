import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    #This difference represents the rotation needed to align the vector ab with the vector bc.
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    #because our hand doesn't rotate 360degree
    if angle >180.0:
        angle = 360-angle
        
    return angle
    
    
    
mp_drawing  =  mp.solutions.drawing_utils # help in visualising poses
mp_pose = mp.solutions.pose# importing pose estimation model



#This line initializes video capture. The 0 argument specifies the first webcam on the system. 
# If you have multiple webcams, you can use 1, 2, etc., to access them. cap = cv2.VideoCapture(0)

#ret, frame = cap.read() Used to read frames 
# Frames=> first image taken series of image taken & shown continously to make a illusion of contionous motion which make a video 
# ret=> true if captures
# frame=> the captured frame

#cv2.waitKey(10): Waits for 10 ms to check for key press. 
# ord('q'): Detects the 'q' key press to exit the loop. 
# cap.release() and cv2.destroyAllWindows(): Clean up resources.

#Video feed
#make detections
#min_detection_confidence: This parameter sets the minimum confidence score required for a body part to be considered detected. 
# If the confidence score for a body part is below this threshold, it won't be detected.
#f the confidence score for a keypoint is high (close to 1), it means the model is very confident that it has accurately detected that keypoint. 
# On the other hand, if the confidence score is low (close to 0), it indicates that the model is uncertain about the detection or estimation of that keypoint.

#min_tracking_confidence: This parameter sets the minimum confidence score required for a detected body part to be included in the pose tracking process. 
# If the confidence score for a body part falls below this threshold during tracking, it won't be considered for further analysis.
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

#curl counter variables
counter = 0
stage = None # describes whether we are at the downpart of our curl or at top part of our curl
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    
    while cap.isOpened():
        ret, frame =  cap.read()
        
        #Recolour the image because by default the image through the webcam will be BGR format not as RGB format so we need to convert
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        #When you set image.flags.writeable = False, you are essentially telling Python that the data in the image array should not be modified. 
        #This allows Python to optimize memory usage and avoid making unnecessary copies of the data when passing it to other functions or libraries.
        
        
        #make  detections
        results = pose.process(image)
        
        # recoloring image to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract the landmarks
        try:
            #EXTRACTING ALL THE LANDMARKS INTO LAANDMARKS LIST
            #mp_pose.PoseLandmark.NOSE.value => 0
            #landmark[mp_pose.PoseLandmark.NOSE.value] => gives the coordianates of nose like x,y,z, visiblity
            landmarks = results.pose_landmarks.landmark
            
            # calculate angle b/w shoulder, elbow, wrist
            #find the angle between left shoulder(11) and wrists(15) to get angle @left elbow(13)
            #get the coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            #find the angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            #visualize angle
            #here first we pass the image we are working on, then the angles, then [640,480] the dimensions of image coming out of my webcam
            #multiply the elbow coordinates by dimensions of image as they are initally in normalise form(0 to 1)
            # convert into int and then into tuple as it is required by cv
            #then we passs the font we want, size of the text, color of text, line width then align
            
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            if angle>160:
                stage = "down"
            if angle < 30  and stage == "down":
                stage = "up"
                counter += 1
                print(counter)
            
            
        except:
            pass
        
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
        #render the detections
        #This line draws the detected pose landmarks on the image using the mp_drawing.draw_landmarks() function.
        # The POSE_CONNECTIONS argument specifies that connections between landmarks should also be drawn.
        # mp_drawing.DrawingSpec(color = (245,117,66), thickness = 2, circle_radius  = 2) => changing the apperarance as we require
        mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_pose.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color = (245,117,66), thickness = 2, circle_radius  = 2),# color of different dots
                                  mp_drawing.DrawingSpec(color = (245,66,230), thickness = 2, circle_radius  = 2)# color of conncetion
                                  )
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()