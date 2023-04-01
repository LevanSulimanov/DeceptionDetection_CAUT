import os
import traceback

# for visual data:
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp

# for audio / video extraction, conversion:
import pydub
import librosa
import scipy
import moviepy.editor as moviepyEditor
from pydub import AudioSegment
from scipy.fftpack import dct  # , fft

# for visualization:
from matplotlib import pyplot as plt



def read_mp3_as_array(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return np.float32(y) / 2**15, a.frame_rate
    else:
        return y, a.frame_rate



#------------------------------------------------------------------------------------
def extract_audio_from_video(video_path, expected_mfcc_feature_num=None):  # , save_mp3_audio_as):
    
    '''
    https://arxiv.org/pdf/1712.04415.pdf uses the following MFCC extraction procedure:
    1. First estimate the periodogram of the power spectrum for each short frame,
    2. Then warp to a Mel frequency scale, and
    3. Finally compute the DCT of the log-Mel-spectrum.
    '''
    
    # tmp file to store for each clip:
    tmp_mp3_path = os.path.join(os.getcwd(), "tmp.mp3")    
    # extract audio from video file:
    clip = moviepyEditor.VideoFileClip(video_path)
    # and save to tmp file
    clip.audio.write_audiofile(tmp_mp3_path)
    
    if expected_mfcc_feature_num is None:
        # read audio into numpy array and also retrieve audio sample rate:
        signal, fs = read_mp3_as_array(tmp_mp3_path, normalized=False)
        print(signal.shape)
        print(fs)

        # 1. estimate the periodogram of the power spectrum for each short frame (PSD)
        # f contains the frequency components
        # S is the PSD
        f, S = scipy.signal.periodogram(signal, fs, scaling='density')

        # 2. warp to a Mel frequency scale
        warped_under_Mel_frequency_scale = np.log(S)

        # 3. compute the DCT of the log-Mel-spectrum
        dct_data = dct(warped_under_Mel_frequency_scale)
        return dct_data
        
    else:
        x, sample_rate = librosa.load(tmp_mp3_path, res_type="kaiser_fast")
        # extract features from the audio:
        dct_data = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=expected_mfcc_feature_num).T, axis=0)
        return dct_data
    
    # remove temporary file:
    while os.path.isfile(tmp_mp3_path):
        try:
            os.remove(tmp_mp3_path)
        except:
            print("\n***ERROR***\n")
            print(traceback.format_exc())



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



#------------------------------------------------------------------------------------
def mkdir_if_none(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



#------------------------------------------------------------------------------------    
def save_label_data(class_num, sample_coordinates_data, sample_audio, save_coord_as, save_audio_as):
    
    class_label_npy = np.array([class_num])
    # sample_coordinates_data
    
    # save_label_as = f"{save_as}_label.npy"
    save_coordinates_as = f"{save_coord_as}_MP_coord.npy"
    save_audio_as = f"{save_audio_as}_audio.npy"
    
    # np.save(save_label_as, class_label_npy)
    np.save(save_coordinates_as, sample_coordinates_data)
    np.save(save_audio_as, sample_audio)
    
    print("Saved with shape:")
    # print("class_label_npy.shape:", class_label_npy.shape)
    print("sample_coordinates_data.shape:", sample_coordinates_data.shape)
    print("sample_audio.shape:", sample_audio.shape)



#------------------------------------------------------------------------------------
# run through all videos and collect - (1) face features and (2) audio features, and (3) save label reference if needed:
def train_data_preparation(video_data_dir="data/Video_chunks/Video_chunks",
                           csv_path="data/Video_chunks/Labels.xlsx",
                           class_to_num = {"truth": 0, "lie": 1},
                           num_to_class = {0: "truth", 1: "lie"},
                           save_keypoints_npy_to=os.path.join(os.getcwd(), "mediaPipe_keypoints_data_UPD"),
                           save_audio_npy_to=os.path.join(os.getcwd(), "audio_data_UPD")):
    
    # make sure to have save folder available:
    mkdir_if_none(save_keypoints_npy_to)
    mkdir_if_none(save_audio_npy_to)
    
    # read train data:
    csv_data = pd.read_excel(csv_path, sheet_name="All_Gestures_Deceptive and Trut")
    
    # go over each video and collect face features + audio features:
    for video_name in os.listdir(video_data_dir):
        # collect information about video:
        class_name = video_name.split("_")[1]
        class_num = class_to_num[class_name]
        video_path = os.path.join(video_data_dir, video_name)
        sample_name = video_name.split(".")[0]
        save_current_coord_as = os.path.join(save_keypoints_npy_to, sample_name)
        save_current_audio_as = os.path.join(save_audio_npy_to, sample_name)
        
        # run through frames of current video:
        extracted_keypoints_npy = process_video(video_path=video_path, class_name=class_name, verbose=True)
        
        # extract audio:
        extracted_audio_npy = extract_audio_from_video(video_path=video_path)  # , save_mp3_audio_as="sample_audio.mp3")
        
        # save training sample:
        save_label_data(class_num=class_num, 
                        sample_coordinates_data=extracted_keypoints_npy, sample_audio=extracted_audio_npy, 
                        save_coord_as=save_current_coord_as,
                        save_audio_as=save_current_audio_as)



# run video processing for (face & audio) feature collection:
train_data_preparation()