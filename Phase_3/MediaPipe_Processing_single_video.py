import os
import cv2
import traceback
import numpy as np
import mediapipe as mp


#------------------------------------------------------------------------------------
def retrieve_face_coordinates_from_frame(image, face_mesh, verbose=True):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    '''
    Collection of detected/tracked faces, where each face is represented as a list of 468 face landmarks and 
    each landmark is composed of x, y and z. x and y are normalized to [0.0, 1.0] by the image width and height 
    respectively. z represents the landmark depth with the depth at center of the head being the origin, 
    and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same
    scale as x.
    '''
    if results.multi_face_landmarks:
        current_set_of_coordinates = []
        for face in results.multi_face_landmarks:
            for landmark in face.landmark:
                x = landmark.x
                y = landmark.y
                z = landmark.z

                current_set_of_coordinates.append(np.array([x,y,z]))

                if verbose:
                    shape = image.shape
                    relative_x = int(x * shape[1])
                    relative_y = int(y * shape[0])

                    cv2.circle(image, (relative_x, relative_y), radius=1, color=(225, 0, 100), thickness=1)
            break  # >>> do only 1 face

            ####################################################################################################
            if len(current_set_of_coordinates) < 478:
                add_more = [np.array([0.0, 0.0, 0.0])] * (478 - len(current_set_of_coordinates))
                current_set_of_coordinates += add_more
            current_set_of_coordinates = current_set_of_coordinates[:478]
            ####################################################################################################
    else:
        # print(traceback.format_exc())
        x, y, z = 0.0, 0.0, 0.0
        current_set_of_coordinates = [np.array([x,y,z])]*478
        
    return image, np.array(current_set_of_coordinates)



#------------------------------------------------------------------------------------
# run through single video:
def process_video(video_path, class_name, webcam=False, verbose=False):
    
    detected_keypoints_coordinates = []
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Media Pipe Initialization:
    # mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    print("\n----------------------\nVideo Capture Path:", video_path)
    
    # For webcam input:
    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(video_path)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.8) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                if webcam:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                else:
                    break
            
            # get coordinates from current frame:
            annotated_face, current_set_of_coordinates = retrieve_face_coordinates_from_frame(image, face_mesh)
            
            # Append detected_keypoints_coordinates:
            detected_keypoints_coordinates.append(np.array(current_set_of_coordinates))
            
            if verbose:
                cv2.imshow('MediaPipe Face Mesh', cv2.flip(annotated_face, 1))
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return np.array(detected_keypoints_coordinates)