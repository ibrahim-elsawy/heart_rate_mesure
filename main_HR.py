import cv2
import numpy as np
import time
import sys
from imutils import face_utils
from face_utilities import Face_utilities
from signal_processing import Signal_processing

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg




def calcHR(video=True, dir=None):

    '''
    Calculates the heart rate of video or live stream

    Args:
        video (bool): false for live stream and true for video
        dir (str): directory of the video
    Outputs:
        lbpm (float): the heart rate of the person
    '''
    if video == False:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(dir)
    
    fu = Face_utilities()
    sp = Signal_processing()
    
    i=0
    last_rects = None
    last_shape = None
    last_age = None
    last_gender = None
    
    face_detect_on = False
    age_gender_on = False

    t = time.time()
    
    #for signal_processing
    BUFFER_SIZE = 100
    
    fps=0 #for real time capture
    video_fps = cap.get(cv2.CAP_PROP_FPS) # for video capture
    
    times = []
    data_buffer = []
    
    # data for plotting
    filtered_data = []
    
    fft_of_interest = []
    freqs_of_interest = []
    
    bpm = 0
    lbpm = []
    
    
    while True:
        # grab a frame -> face detection -> crop the face -> 68 facial landmarks -> get mask from those landmarks

        # calculate time for each loop
        t0 = time.time()
        
        if(i%1==0):
            face_detect_on = True
            if(i%10==0):
                age_gender_on = True
            else:
                age_gender_on = False
        else: 
            face_detect_on = False
        
        ret, frame = cap.read()
        
        if frame is None:
            print("End of video")
            break
        
        
        
        ret_process = fu.no_age_gender_face_process(frame, "68")
        
        if ret_process is None:
            continue
        
        rects, face, shape, aligned_face, aligned_shape = ret_process
        
           
            
        #for signal_processing
        ROIs = fu.ROI_extraction(aligned_face, aligned_shape)
        green_val = sp.extract_color(ROIs)
        
        data_buffer.append(green_val)
        
        if(video==False):
            times.append(time.time() - t)
        else:
            times.append((1.0/video_fps)*i)
        
        L = len(data_buffer)
        
        if L > BUFFER_SIZE:
            data_buffer = data_buffer[-BUFFER_SIZE:]
            times = times[-BUFFER_SIZE:]
            L = BUFFER_SIZE
        if L==100:
            fps = float(L) / (times[-1] - times[0])
            detrended_data = sp.signal_detrending(data_buffer)
            interpolated_data = sp.interpolation(detrended_data, times)
            
            normalized_data = sp.normalization(interpolated_data)
            
            fft_of_interest, freqs_of_interest = sp.fft(normalized_data, fps)
            
            max_arg = np.argmax(fft_of_interest)
            bpm = freqs_of_interest[max_arg]
            lbpm.append(bpm)
            print(f"{bpm}   ***************\n")
            filtered_data = sp.butter_bandpass_filter(interpolated_data, (bpm-20)/60, (bpm+20)/60, fps, order = 3)
            
        i = i + 1        
    cap.release()
    print(f'the average of BPM is   {np.average(np.array(lbpm))}')
    print("total running time: " + str(time.time() - t))
    return np.average(np.array(lbpm))

if __name__ == '__main__':
    print(calcHR(True, r'D:\upwork\Heart-rate-measurement-using-camera\new_update\v1.mp4'))