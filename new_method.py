from pynput.mouse import Button, Controller
import cv2
import tkinter as tk
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from collections import deque

def get_color_signature(cap,hist):
    key = cv2.waitKey(100)
    while key != ord('q')  and key != ord('l'):
        frame_hist = cap.read()[1]
        frame_hist = cv2.flip(frame_hist, 1)
        cv2.rectangle(frame_hist, (377, 307), (423, 373), color=(255, 255, 255), thickness=2)
        working_frame = frame_hist[310:370, 380:420]
        image_hsv = cv2.cvtColor(working_frame, cv2.COLOR_BGR2HSV)
        hist_temp = cv2.calcHist([image_hsv], [0, 1], None, [256, 256], (0, 256, 0, 256))
        #image_hsv = cv2.cvtColor(working_frame, cv2.COLOR_BGR2YCR_CB)
        #hist_temp = cv2.calcHist([image_hsv], [ 0, 1, 2], None, [ 256, 256, 256], ( 0, 256, 0, 256, 0, 256))
        frame_hist[80:336,0:256] = cv2.cvtColor(hist_temp,cv2.COLOR_GRAY2BGR) * 255
        frame_hist[0:80, :] = (255,255,255)
        frame_hist = cv2.putText(frame_hist, "Place hand in the white box. Then", (0, 10), cv2.FONT_HERSHEY_PLAIN,fontScale=1, thickness=1, color=(0, 0, 0))
        frame_hist = cv2.putText(frame_hist, "press 'r' to record current picture, ", (0, 20), cv2.FONT_HERSHEY_PLAIN,fontScale=1, thickness=1, color=(0, 0, 0))
        frame_hist = cv2.putText(frame_hist, "press 'q' to quit or" , (0, 30), cv2.FONT_HERSHEY_PLAIN,fontScale=1, thickness=1, color=(0, 0, 0))
        frame_hist = cv2.putText(frame_hist, "press 'c' to clear the histogram's history.", (0, 40), cv2.FONT_HERSHEY_PLAIN,fontScale=1, thickness=1, color=(0, 0, 0))
        frame_hist = cv2.putText(frame_hist, "press 's' to save current histogram to disk.", (0, 40),
                                 cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(0, 0, 0))
        frame_hist = cv2.putText(frame_hist, "press 'l' to reload the histogram from disk.", (0, 40),
                                 cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(0, 0, 0))
        frame_hist = cv2.putText(frame_hist, 'Histogram (Hue, saturation)', (0, 70), cv2.FONT_HERSHEY_PLAIN,
                                 fontScale=1, thickness=2, color=(0, 0, 0))
        cv2.imshow('color_signature', frame_hist)
        key = cv2.waitKey(20)
        if key == ord('r'):
            if hist is not None:
                hist = hist + hist_temp
                cv2.imshow('color_signature_hist', hist)
            else:
                hist = hist_temp
        if key == ord('c'):
            hist = None
        if key == ord('l'):
            hist = np.fromfile('.\hist.np',sep=",").reshape(256,256)
            cv2.imshow('color_signature_hist', hist)
        if key == ord('s'):
            if hist is not None:
                hist.tofile('.\hist.np', sep=",")
            else:
                print("Can't save to file. hist is not recorded yet.")
    cv2.destroyWindow('color_signature')
    cv2.destroyWindow('color_signature_hist')
    return hist

def control_by_method(positions,mouse_action = 'move', SquareSpeed=False, mb_right = False, mb_left=False, DispOnly=False, mb_left_double_click=False):
    if mouse_action == 'move':
        if SquareSpeed:
            x = int(positions['disp'][0] * positions['motion_speed'][0] * control_parameters['control_speed'])
            y = int(positions['disp'][1] * positions['motion_speed'][1] * control_parameters['control_speed'])
        if DispOnly:
            x = int(positions['disp'][0])
            y = int(positions['disp'][1])
        mouse.move(x, y)
        return (x, y)

mouse = Controller()
cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorKNN()
hist =None
hist = get_color_signature(cap,hist)
control_parameters = {}
filter_parameters = {}
control_parameters['control_speed'] = 0.006

top = tk.Tk()


positions = {
        'disp' : np.array([0, 0]),
        'proc_time' : 0,
        'top_mean' : (0,0),
        'hull_matching_index':0.05
    }

ret,frame = cap.read()
# setup initial location of window
x, y, w, h = 300, 200, 100, 200 # simply hardcoded the values
track_window_1 = (x, y, w, h)
track_window_2 = (100, 200, 100, 100)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
results = [np.array([240,320])]
learning = True
results_tw = []
motion_stable_tracker = deque(np.array([0] * 10))
while(1):
    ret, frame = cap.read()
    positions['proc_time'] = time.time()
    frame = cv2.flip(frame,1)
    frame = frame[60:420]

    if ret == True:
        if learning:
            fgMask = backSub.apply(frame,learningRate=0.8)
        else:
            fgMask = backSub.apply(frame)
        morphed_open = cv2.morphologyEx(fgMask,cv2.MORPH_OPEN,np.ones((3,3)))
        morphed_close = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, np.ones((3, 3)))
        maskd = cv2.bitwise_and(frame, frame, mask=morphed_open)
        hsv = cv2.cvtColor(maskd, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0,1],hist,[0,256,0,256],255)
        # apply camshift to get the new location
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        ret, track_window_1 = cv2.CamShift(dst, track_window_1, term_crit)
        #ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        results.append(np.array(ret[0]))
        positions['proc_time'] = time.time() - positions['proc_time'] + 0.001 * 10
        dist = np.sqrt(np.sum((np.int32(results[-1]) - np.int32(results[-2])) ** 2))
        dist_elem = results[-1] - results[-2]
        x, y, w, h = track_window_1
        #img2_track = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame.copy(), [pts], True, 255, 2)
        #img2 = cv2.drawMarker(img2, tuple(np.int32(ret[0])), 255)
        img2 = cv2.putText(img2, "1", tuple(np.int32(ret[0])), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1,
                    color=(0, 255, 0))

        motion_stable_tracker.append(ret[1][0]*ret[1][1])
        motion_stable_tracker.popleft()
        stable_motion = np.std(motion_stable_tracker) < 25000
        motion_detected = 0 < ret[1][0]*ret[1][1] < 40000
        results_tw.append(np.std(motion_stable_tracker))
        if dist <= 200:
            positions['disp'] = dist_elem
            positions['motion_speed'] = np.abs(positions['disp'] / positions['proc_time'])
            if stable_motion and motion_detected:
                wide_window = (ret[0], (ret[1][0] + 50, ret[1][1] + 50), ret[2])
                pts = cv2.boxPoints(wide_window)
                pts = np.int0(pts)
                img3 = cv2.polylines(frame.copy(), [pts], True, 255, 2)
                img3 = cv2.rectangle(img3, (x - 30, y - 30), (x + w + 30, y + h + 30), (0, 0, 255), 2)
                cv2.imshow('img3', img3)
                pass
                #control_by_method(positions, SquareSpeed=True, mouse_action='move')
        else:
            print("skipping ",dist)
        # center_difference = np.abs(np.array(track_window_2[0:2]) - np.array(track_window_1[0:2]))
        # if np.sum(center_difference) < 100:
        #     print("together")
        #     track_window_2 = (track_window_2[0] - 20,track_window_2[1] - 20,track_window_2[2],track_window_2[3])
        # #    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 100)
        # else:
        #     print("seperate")
        # ret2, track_window_2 = cv2.CamShift(dst, track_window_2, term_crit)
        # pts = cv2.boxPoints(ret2)
        # pts = np.int0(pts)
        # img2 = cv2.polylines(img2, [pts], True, (255,255,0), 2)
        # img2 = cv2.drawMarker(img2, tuple(np.int32(ret2[0])), 255)
        # img2 = cv2.putText(img2, "2", tuple(np.int32(ret2[0])), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1,
        #                    color=(0, 255, 0))
        if stable_motion:
            if motion_detected:
                img2 = cv2.putText(img2,"Motion Detected",(0, 30), cv2.FONT_HERSHEY_PLAIN,fontScale=1, thickness=1, color=(0, 255, 0))
            else:
                img2 = cv2.putText(img2, "No Motion Detected", (0, 30), cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                                         thickness=1, color=(0, 255, 0))
        else:
            img2 = cv2.putText(img2, "No Stable Motion Detected", (0, 30), cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                                     thickness=1, color=(0, 255, 0))

        cv2.imshow('fgMask',fgMask)
        cv2.imshow('morphed_open', morphed_open)
        cv2.imshow('morphed_close', morphed_close)
        cv2.imshow('maskd',maskd)
        cv2.imshow('dst', dst)
        cv2.imshow('img2',img2)
        # cv2.imshow('img2_track', img2_track)
    else:
        break
    k = cv2.waitKey(10) & 0xff
    if k == ord('l'):
        learning = not learning
        print("learning: ", learning)
    if k == 27:
        break


class DetectionInstance:
    def __init__(self,frame,fgmask,fgmask_morphed,maskd,dst,cur_track_window,next_track_window,ret,img2):
        self.frame = frame
        self.fgmask = fgmask
        self.fgmask_morphed = fgmask_morphed
        self.frame_filtered_background = maskd
        self.frame_filtered_full = dst
        self.cur_track_window = cur_track_window
        self.next_track_window = next_track_window
        self.detected_object_center = ret[0]
        self.detected_object_width = ret[1][0]
        self.detected_object_height = ret[1][0]
        self.detected_object_orientation = ret[2]
        self.final_image_result = img2

#
stream = []
backSub = cv2.createBackgroundSubtractorKNN()
for i in np.arange(100):
    ret1, frame = cap.read()
    frame = cv2.flip(frame,1)
    fgMask = backSub.apply(frame)
    morphed_open = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, np.ones((3, 3)))
    maskd = cv2.bitwise_and(frame, frame, mask=morphed_open)
    hsv = cv2.cvtColor(maskd, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 256, 0, 256], 255)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    cur_track_window = track_window_1
    ret, track_window_1 = cv2.CamShift(dst, track_window_1, term_crit)
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(frame.copy(), [pts], True, 255, 2)
    img2 = cv2.drawMarker(img2, tuple(np.int32(ret[0])), 255)
    cv2.imshow('image1', frame)
    cv2.imshow('fgMask', fgMask)
    cv2.imshow('img2', img2)
    stream.append(DetectionInstance(frame,fgMask,morphed_open,maskd,dst,cur_track_window,track_window_1,ret,img2))
    cv2.waitKey(60) & 0xff
#
# for i in np.arange(100):
#     frame, fgmask, img2 = stream[i].frame,stream[i].fgmask,stream[i].final_image_result
#     cv2.imshow('image1', frame)
#     cv2.imshow('fgMask', fgMask)
#     cv2.imshow('img2', img2)
#     cv2.waitKey(60) & 0xff
#
# cv2.imshow('image1',stream[-5])
# cv2.imshow('image2',stream[-6])