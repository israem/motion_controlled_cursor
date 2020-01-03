from pynput.mouse import Button, Controller
import numpy as np
import cv2, ctypes
import time
from collections import deque
import tkinter as tk
import os


DEBUG = False
cap = cv2.VideoCapture(0)
mouse = Controller()

control_parameters = {}
video_parameters = {}
filter_parameters = {}
gesture_control = {}
top = tk.Tk()


hist = None


def alloc_camera():
    return cv2.VideoCapture(0)


def close_all(cap):
    cap.release()
    cv2.destroyAllWindows()


def show_images(images, titles):
    for i, e in zip(images,titles):
        cv2.imshow('Live video (' + e + ') 4', i)

def display_image(frame,title,waittime = 1):
    for i in range(4):
        show_images([frame],[title])
        cv2.waitKey(waittime)


def set_parameter(value, key, parameters, fraction=1.0, offset = 0):
    if type(parameters[key]) == int:
        test_val = value / fraction + offset
        parameters[key] = int(test_val)
    if type(parameters[key]) == str:
        parameters[key] = str(value)
    if type(parameters[key]) == float:
        test_val = value / fraction + offset
        parameters[key] = float(test_val)
    if type(parameters[key]) == bool:
        parameters[key] = value


def quit():
    control_parameters['run'] = False


def get_color_signature():
    global hist
    images = []
    key = cv2.waitKey(100)
    while key != ord('r') and key != ord('a'):
        frame_hist = cap.read()[1]
        frame_hist = cv2.flip(frame_hist, 1)
        cv2.rectangle(frame_hist, (377, 307), (423, 373), color=(255, 255, 255), thickness=2)
        show_images([frame_hist], ['color_signature'])
        key = cv2.waitKey(100)
        if key == ord('r') or key == ord('a'):
            frame_hist = cap.read()[1]
            frame_hist = cv2.flip(frame_hist, 1)
            working_frame = frame_hist[310:370, 380:420]
            imageHSV = cv2.cvtColor(working_frame, cv2.COLOR_BGR2HSV)
            images.append(imageHSV)
    if key == ord('r'):
        hist = cv2.calcHist(images, [0, 1], None, [256, 256], ( 0, 256, 0, 256))
    else:
        hist = cv2.calcHist(images, [0, 1], None, [256, 256], ( 0, 256, 0, 256), hist)
    cv2.destroyWindow('color_signature')


def draw_motion_frame(frame_motion_subframe, image_dimensions):
    frame_motion_subframe = cv2.line(frame_motion_subframe, (image_dimensions[2], image_dimensions[0]),
                                     (image_dimensions[3], image_dimensions[0]), (255, 40, 255), thickness=2)
    frame_motion_subframe = cv2.line(frame_motion_subframe, (image_dimensions[2], image_dimensions[1]),
                                     (image_dimensions[3], image_dimensions[1]), (255, 80, 255), thickness=2)
    frame_motion_subframe = cv2.line(frame_motion_subframe, (image_dimensions[2], image_dimensions[0]),
                                     (image_dimensions[2], image_dimensions[1]), (255, 120, 255), thickness=2)
    frame_motion_subframe = cv2.line(frame_motion_subframe, (image_dimensions[3], image_dimensions[0]),
                                     (image_dimensions[3], image_dimensions[1]), (255, 160, 255), thickness=2)
    show_images([frame_motion_subframe], ["motion sub frame"])



def save_image(thresh):
    export_frame = thresh.copy()
    columns_to_add = int(export_frame.shape[0] / 0.375 - export_frame.shape[1])
    left_columns = np.zeros((export_frame.shape[0], columns_to_add // 2))
    right_columns = np.zeros((export_frame.shape[0], columns_to_add // 2))
    img_new = np.hstack([left_columns, export_frame, right_columns])
    img_new = cv2.resize(img_new, (640, 240))
    print("saving to " + control_parameters['image_label'])
    dirname = ('C:\\Users\\isrmi\\OneDrive\\school_spring_2019\\dsp_lab\\lecture 6\\demo 20 - video _cv2_\\demo 20 - video (cv2)\\5 - video operations\\images\\' + str(control_parameters['image_label']) + '\\')
    try:
        os.mkdir((dirname))
    except:
        pass
    cv2.imwrite(dirname + str(time.time()) + '.png',img_new)

def calc_motion_contours(stream):
    kernel = np.ones((video_parameters['kernel_size_full_frame'], video_parameters['kernel_size_full_frame']), np.uint8)
    frame_xor = cv2.cvtColor(stream[-2], cv2.COLOR_BGR2GRAY)
    frame_xor = cv2.absdiff(frame_xor, cv2.cvtColor(stream[-1], cv2.COLOR_BGR2GRAY))
    frame_eroded = cv2.erode(frame_xor, kernel, iterations=1)
    frame_morphred = cv2.morphologyEx(frame_eroded, cv2.MORPH_OPEN, kernel)
    ret, thresh = cv2.threshold(frame_morphred, video_parameters['threshhold_full_frame'], 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, frame_xor, frame_eroded, frame_morphred, thresh


def calc_motion_frame_contours(frame, image_dimensions):
    working_frame = frame[image_dimensions[0]:image_dimensions[1], image_dimensions[2]:image_dimensions[3]]
    # Get pointer to video frames from primary device
    imageHSV = cv2.cvtColor(working_frame.copy(), cv2.COLOR_BGR2HSV)
    if hist is None:
        min_HSV = np.array([0, 58, 30])
        max_HSV = np.array([33, 255, 255])
        skinRegionHSV = cv2.inRange(imageHSV, min_HSV, max_HSV)
    else:
        skinRegionHSV = cv2.calcBackProject([imageHSV], [0, 1], hist, (0, 256, 0, 256), 1)
    skinHSV = cv2.bitwise_and(working_frame, working_frame, mask=skinRegionHSV)
    frame_xor = cv2.cvtColor(cv2.cvtColor(skinHSV, cv2.COLOR_HSV2BGR),cv2.COLOR_BGR2GRAY)
    kernel = np.ones((video_parameters['kernel_size_focus_frame'], video_parameters['kernel_size_focus_frame']), np.uint8)
    ret, thresh = cv2.threshold(frame_xor, video_parameters['threshhold_focus_frame'], 255, cv2.THRESH_BINARY)
    frame_dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(frame_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(image_dimensions[2], image_dimensions[0]))
    # filters the countours. only consider contours with more points than parameters['min_contour_length']
    contours = [con for con in contours if len(con) > filter_parameters['min_contour_length']]
    return contours, working_frame, frame_dilated


def calc_segments(contours,positions):
    positions['num_countours'] = len(contours)
    if positions['num_countours'] > 0:
        points_left = np.array([con[np.argmin(con[:, 0, 0])].ravel() for con in contours])
        points_up = np.array([con[np.argmin(con[:, 0, 1])].ravel() for con in contours])
        points_right = np.array([con[np.argmax(con[:, 0, 0])].ravel() for con in contours])
        points_down = np.array([con[np.argmax(con[:, 0, 1])].ravel() for con in contours])
        positions['top'] = points_up[np.argmin(points_up[:, 1])]
        positions['left'] = points_left[np.argmin(points_left[:, 0])]
        positions['right'] = points_right[np.argmax(points_right[:, 0])]
        positions['down'] = points_down[np.argmax(points_down[:, 1])]
        positions['down_mean'] = ([np.mean([positions['left'][0], positions['right'][0]]).astype(int),
                               2*np.mean([positions['left'][1], positions['right'][1]]).astype(int) - positions['top'][1] ])


def calc_motion_frame_dimensions(segment_moving,segment_stationary):
    temp_image_dimensions = [-video_parameters['focus_frame_margin'], video_parameters['focus_frame_margin'], -video_parameters['focus_frame_margin'], video_parameters['focus_frame_margin']]
    if video_parameters['left_hand']:
        right_points = np.array([segment_moving['right'][0], segment_stationary['right'][0]])
        left_points = np.array([segment_stationary['left'][0]])
    else:
        left_points = np.array([segment_moving['left'][0], segment_stationary['left'][0]])
        right_points = np.array([segment_stationary['right'][0]])
    up_points = np.array([segment_moving['top'][1], segment_stationary['top'][1]])
    down_points = np.array([segment_stationary['down'][1]])
    temp_image_dimensions[0] += np.mean(up_points).astype(int)
    temp_image_dimensions[1] += np.mean(down_points).astype(int)
    temp_image_dimensions[2] += np.mean(left_points).astype(int)
    temp_image_dimensions[3] += np.mean(right_points).astype(int)
    ratio = (temp_image_dimensions[1] - temp_image_dimensions[0]) / (
        temp_image_dimensions[3] - temp_image_dimensions[2])
    area_temp = (temp_image_dimensions[1] - temp_image_dimensions[0]) * (
            temp_image_dimensions[3] - temp_image_dimensions[2])
    if temp_image_dimensions[0] < 0:
        temp_image_dimensions[0] = 0
    if temp_image_dimensions[1] > 480:
        temp_image_dimensions[1] = 480
    if temp_image_dimensions[2] < 0:
        temp_image_dimensions[2] = 0
    if temp_image_dimensions[3] > 640:
        temp_image_dimensions[3] = 640
    return ratio, area_temp, temp_image_dimensions

def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None

def control_by_method(positions, SquareSpeed=False, mb_right = False, mb_left=False, DispOnly=False, mb_left_double_click=False):
    point = mouse.position
    if SquareSpeed:
        mouse.move(
            int(positions['disp'][0] * positions['motion_speed'][0] * control_parameters['control_speed']),
            int(positions['disp'][1] * positions['motion_speed'][1] * control_parameters['control_speed'])
        )
    if DispOnly:
        mouse.move(
            int(positions['disp'][0]),
            int(positions['disp'][1])
        )
    if mb_left:
        if time.time() - control_parameters['pressed'] > 0.5:
            control_parameters['pressed'] = time.time()
            mouse.click(Button.left,1)
    if mb_left_double_click:
        if time.time() - control_parameters['pressed'] > 0.5:
            control_parameters['pressed'] = time.time()
            mouse.click(Button.left,2)
    if mb_right:
        if time.time() - control_parameters['pressed'] > 0.5:
            control_parameters['pressed'] = time.time()
            mouse.click(Button.right, 2)

def run(cap = cap):
    image_dimensions = [0, 480, 0, 640]
    centeroid_pt = (0,0)
    stream = deque()
    hull = None
    line_stream = deque(np.array([(0,0),(0,0),(0,0),(0,0)]))
    thumb_mean_stream = deque(np.array([0]*filter_parameters['gesture_detection_filter_size']))
    pinki_mean_stream = deque(np.array([0]*filter_parameters['gesture_detection_filter_size']))
    top_f_mean_stream = deque(np.array([0]*filter_parameters['gesture_detection_filter_size']))
    positions = {
        'disp' : np.array([0, 0]),
        'proc_time' : 0,
        'top_mean' : (0,0),
        'hull_matching_index':0.0
    }
    segment_moving = {
        'top':np.array([0, 240])
    }
    segment_stationary = {
        'top': np.array([0, 0]),
        'left': np.array([0, 0]),
        'right': np.array([640, 0]),
        'down': np.array([0, 480])
    }

    [ok, frame] = cap.read()
    stream.append(frame.copy())
    while ok and control_parameters['run']:
        top.update()

        #process keys
        key = cv2.waitKey(video_parameters['wait_time'])
        if key == ord('s'):
            control_parameters['save_current_image'] = True

        # clear queues
        while len(stream) >= 2:
            stream.popleft()
        while len(thumb_mean_stream) >= filter_parameters['gesture_detection_filter_size']:
            thumb_mean_stream.popleft()
        while len(pinki_mean_stream) >= filter_parameters['gesture_detection_filter_size']:
            pinki_mean_stream.popleft()
        while len(top_f_mean_stream) >= filter_parameters['gesture_detection_filter_size']:
            top_f_mean_stream.popleft()
        while len(line_stream) >= filter_parameters['line_stream_length']:
            line_stream.popleft()

        # grab new frame
        positions['proc_time'] = time.time()
        [ok, frame] = cap.read()
        frame = cv2.flip(frame, 1)
        stream.append(frame.copy())

        # calculate countours and segments - motion
        contours, frame_xor, frame_eroded, frame_morphred, thresh = calc_motion_contours(stream.copy())
        if control_parameters['save_current_image']:
            save_image(frame_xor)
            save_image(frame_eroded)
            save_image(frame_morphred)
            save_image(thresh)
            control_parameters['save_current_image'] = False

        if len(contours) > 0:
            merged_contours_motion = []
            for i, con in enumerate(contours):
                merged_contours_motion.extend(con)
            hull_full_frame = cv2.convexHull(np.array(merged_contours_motion))
            calc_segments([hull_full_frame], segment_moving)

        # calculate countours and segments - sub frame
        contours_sub_Frame, sub_frame, thresh = calc_motion_frame_contours(frame.copy(), image_dimensions)

        if len(contours_sub_Frame) > 0:
            # find centroid from the top contour

            merged_contours = []
            for i, con in enumerate(contours_sub_Frame):
                merged_contours.extend(con)
            if hull is not None:
                temp_hull = hull.copy()
                hull = cv2.convexHull(np.array(merged_contours))
                positions['hull_matching_index'] = cv2.matchShapes(hull, temp_hull, cv2.CONTOURS_MATCH_I3, 0.0)
                if positions['hull_matching_index'] < filter_parameters['shape_matching_threshhold'] or key == 'w':
                    positions['prev_hull'] = temp_hull.copy()
                    positions['hull_matching_index'] = cv2.matchShapes(hull, positions['prev_hull'],cv2.CONTOURS_MATCH_I3, 0.0)
            else:
                hull = cv2.convexHull(np.array(merged_contours))

            # calculate centroid
            centeroid_pt = centroid(hull)
            line_stream.append(centeroid_pt)
            calc_segments([hull], segment_stationary)
            # calculate the new sub frame
            ratio, area_temp, temp_dims = calc_motion_frame_dimensions(segment_moving, segment_stationary)
            if area_temp > filter_parameters['min_area']:
                image_dimensions = temp_dims

            if control_parameters['save_current_image']:

                save_image(thresh)
                control_parameters['save_current_image'] = False

            # This position values are used to adjust the mouse cursor coordinates.
            # Every iteration the mouse will move 'disp' (for displacement) pixels multiplied by the speed
            positions['proc_time'] = time.time() - positions['proc_time'] + 0.001*video_parameters['wait_time']
            positions['prev_top_mean'] = positions['top_mean']
            positions['top_mean'] = tuple(np.mean(np.array(line_stream), axis=0).astype(int))
            positions['disp'] = np.array(positions['top_mean']) - np.array(positions['prev_top_mean'])
            positions['motion_speed'] = np.abs(positions['disp'] / positions['proc_time'])

            # Gesture control, the queues acting as filters to measure the mean distances over time.
            # So we can dinamically detect whether a significat motion was done if the current is much larger then the mean
            # since every new value continuously fed into the filter the mean value will adjust itself,
            # essentialy allowing the gesture to be performed only once over time.
            thumb_mean_stream.append((np.array(centeroid_pt) - segment_stationary['left'])[0])
            positions['thumb_mean'] = np.sum(np.array(thumb_mean_stream)) // filter_parameters['gesture_detection_filter_size']
            pinki_mean_stream.append((segment_stationary['right'] - np.array(centeroid_pt))[0])
            positions['pinki_mean'] = np.sum(np.array(pinki_mean_stream)) // filter_parameters['gesture_detection_filter_size']
            top_f_mean_stream.append((np.array(centeroid_pt) - segment_stationary['top'])[1])
            positions['top_f_mean'] = np.sum(np.array(top_f_mean_stream)) // filter_parameters['gesture_detection_filter_size']

            if control_parameters['control']:
                if (np.array(centeroid_pt) - segment_stationary['left'])[0] > 1.8*positions['thumb_mean']:
                    gesture_control['thumb'] = True
                if (segment_stationary['right'] - np.array(centeroid_pt))[0] > 1.8 * positions['pinki_mean']:
                    gesture_control['pinki'] = True
                if (np.array(centeroid_pt) - segment_stationary['top'])[1] < 0.6 * positions['top_f_mean']:
                    gesture_control['top'] = True

                if gesture_control['thumb'] and gesture_control['pinki'] and gesture_control['top'] and len(contours) > 0:
                    control_by_method(positions, mb_left=True)
                    gesture_control['thumb'] = False
                    gesture_control['top'] = False
                    gesture_control['pinki'] = False
                if positions['hull_matching_index'] < filter_parameters['shape_matching_threshhold']:
                    control_by_method(positions, SquareSpeed=True)
        frame_conts = cv2.drawContours(frame.copy(), contours_sub_Frame, -1, (0, 255, 0), 1)
        frame_conts = cv2.drawContours(frame_conts, [hull], -1, (255, 0, 0), 1)
        frame_conts = cv2.drawMarker(frame_conts, centeroid_pt,(255,255,255),cv2.MARKER_CROSS, thickness=2 )
        frame_conts = cv2.drawMarker(frame_conts, tuple(segment_stationary['top']), (255, 255, 255), cv2.MARKER_CROSS, thickness=2)
        frame_conts = cv2.drawMarker(frame_conts, positions['top_mean'], (0,0,0), cv2.MARKER_CROSS, thickness=2)
        draw_motion_frame(frame_conts, image_dimensions)




video_parameters['kernel_size_full_frame'] = 3
video_parameters['kernel_size_focus_frame'] = 3
video_parameters['wait_time'] = 1
video_parameters['threshhold_focus_frame'] = 50
video_parameters['threshhold_full_frame'] = 20
video_parameters['focus_frame_margin'] = 90
video_parameters['left_hand'] = False

filter_parameters['line_stream_length'] = 3
filter_parameters['min_contour_length'] = 90
filter_parameters['min_area'] = 10000
filter_parameters['gesture_detection_filter_size'] = 30
filter_parameters['shape_matching_threshhold'] = 0.1

control_parameters['pressed'] = time.time()
control_parameters['control_speed'] = 0.002
control_parameters['run'] = True
control_parameters['control'] = False
control_parameters['save_current_image'] = False
control_parameters['image_label'] = ''

gesture_control['thumb'] = False
gesture_control['pinki'] = False
gesture_control['top'] = False

scales = {}
keys = ['kernel_size_full_frame', 'kernel_size_focus_frame','wait_time','threshhold_full_frame','threshhold_focus_frame','focus_frame_margin','shape_matching_threshhold',
        'line_stream_length','min_contour_length','control_speed']
for i, key in enumerate(keys):
    scales[key] = tk.Scale(top, label = key, length=300, from_ = 1, to = 10, tickinterval = 1, orient='horizontal')
    if i >= (len(keys)//2):
        scales[key].grid(row=i - (len(keys)//2), column=4, columnspan=4, rowspan=1)
    else:
        scales[key].grid(row=i, column=0, columnspan=4, rowspan=1)


scales['kernel_size_full_frame'].config( command = lambda x : set_parameter(int(x),'kernel_size_full_frame',video_parameters))
scales['kernel_size_focus_frame'].config( command = lambda x : set_parameter(int(x),'kernel_size_focus_frame',video_parameters))
scales['wait_time'].config(to= 20,              command = lambda x : set_parameter(int(x),'wait_time',video_parameters,fraction=0.2))
scales['threshhold_full_frame'].config(to= 25 , command = lambda x : set_parameter(int(x),'threshhold_full_frame',video_parameters,fraction=0.1))
scales['threshhold_focus_frame'].config(to= 25 , command = lambda x : set_parameter(int(x),'threshhold_focus_frame',video_parameters,fraction=0.1))

scales['focus_frame_margin'].config(to= 20,                 command = lambda x : set_parameter(int(x),'focus_frame_margin',video_parameters,fraction=0.2))

scales['shape_matching_threshhold'].config(to= 20 , command = lambda x : set_parameter(int(x),'shape_matching_threshhold',filter_parameters,fraction=20))
#scales['line_stream_length'].config(from_=1,to=10, command = lambda x : set_parameter(int(x),'line_stream_length',filter_parameters))
scales['min_contour_length'].config(to= 25,     command = lambda x : set_parameter(int(x),'min_contour_length',filter_parameters,fraction=0.1))
scales['control_speed'].config(                         command = lambda x : set_parameter(int(x),'control_speed',control_parameters,fraction=1000))

scales['kernel_size_full_frame'].set(3)
scales['kernel_size_focus_frame'].set(3)
scales['wait_time'].set(1)
scales['threshhold_full_frame'].set(2)
scales['threshhold_focus_frame'].set(7)
scales['focus_frame_margin'].set(9)
scales['line_stream_length'].set(2)
scales['min_contour_length'].set(9)
scales['control_speed'].set(6)



iamge_label_txb = tk.Entry(top)
iamge_label_txb.bind("<Return>",lambda x : set_parameter(iamge_label_txb.get(),control_parameters, 'image_label'))
iamge_label_txb.grid(row=0,column=8)
get_color_signature_button = tk.Button(top, text="Get Color Signature",command=get_color_signature)
get_color_signature_button.grid(row=1, column= 8)
control_checkbox = tk.Checkbutton(top, text="control", command= lambda: set_parameter(not control_parameters['control'],'control',control_parameters))
control_checkbox.grid(row=2, column=8)

quit_button = tk.Button(top, text="Quit",command=quit)
quit_button.grid(row=3, column= 8)


run()