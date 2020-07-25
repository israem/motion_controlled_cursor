import PIL.Image, PIL.ImageTk
from pynput.mouse import Button, Controller
import numpy as np
import cv2
import time
from collections import deque
import tkinter as tk
import os

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

DEBUG = False
print('Connecting to camera..')
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('.\\video.mp4',cv2.VideoWriter_fourcc(*'XVID'), 25, (640,480))
mouse = Controller()
print('Done!')
control_parameters = {}
video_parameters = {}
filter_parameters = {}
gesture_control = {}
top = tk.Tk()


hist = None
hull = None

def tutorial():
    tutorials_pages = []
    tutorials_pages.append('')
    tutorial_window = tk.Tk()
    tutorial_window.title = "Tutorial"
    def display_frame(frame):
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame), master=tutorial_window)
        canvas.config(image=photo)
        canvas.image=photo
        tutorial_window.update()
    def play_video(video):
        wait = int(1 // video.get(cv2.CAP_PROP_FPS))
        ok, video_frame = video.read()
        while ok:
            ok, video_frame = video.read()
            video_frame = cv2.flip(video_frame, 1)
            display_frame(video_frame)
            key = cv2.waitKey(wait)
    first_frame = np.ones((480,640,3)).astype(np.uint8)*255
    canvas = tk.Label(tutorial_window, width=640, height=480)
    canvas.grid(row= 0, column = 0, columnspan=3)
    back_button = tk.Button(tutorial_window,text='< Back')
    back_button.grid(row=1, column=0)
    next_button = tk.Button(tutorial_window, text='Next >', command = lambda : play_video(cap))
    next_button.grid(row=1, column=1)
    quit_button = tk.Button(tutorial_window, text='quit', command=tutorial_window.destroy)
    quit_button.grid(row=1, column=2)
    first_frame = cv2.putText(first_frame, 'Tutorial',org = (40,40),color=(0,0,0),fontScale=1,fontFace=cv2.FONT_HERSHEY_PLAIN)
    first_frame = cv2.putText(first_frame, 'Something', org=(40, 60), color=(0, 0, 0), fontScale=1,fontFace=cv2.FONT_HERSHEY_PLAIN)
    display_frame(first_frame)
    tutorial_window.update()


class CentroidFilter:
    def __init__(self, name, defualt_value=0, filter_length=30, match_factor = 30, change_threshold = 20, defaults_to = True):
        self.stream_out = deque(np.array([defualt_value] * filter_length))
        self.stream_in = deque(np.array([defualt_value] * filter_length))
        self.filter_length = filter_length
        self.mean_out = 0
        self.mean_in = 0
        self.gesture_name = name
        self.gesture_in = False
        self.gesture_out = False
        self.match_factor = match_factor
        self.change_threshold = change_threshold
        self.defaults_to = defaults_to

    def __repr__(self):
        return self.gesture_name + ": " + str(self.gesture_out) + " - " + str(int(self.mean_in)) + " - " + str(int(self.mean_out)) + " - " + str(self.last_diff)

    def __str__(self):
        return self.gesture_name + ": " + str(self.gesture_out) + " - " + str(int(self.mean_in)) + " - " + str(int(self.mean_out)) + " - " + str(self.last_diff)

    def update_means(self):
        self.mean_out = np.mean(np.array(self.stream_out))
        self.mean_in = np.mean(np.array(self.stream_in))

    def add(self, value_low, value_high):
        self.last_diff = np.abs(value_high - value_low)
        self.stream_in.append(self.last_diff)
        self.stream_out.append(self.last_diff)
        self.update_means()
        self.test(value_high, value_low)

    def release_points(self):
        while len(self.stream_in) >= self.filter_length:
            self.stream_in.popleft()
        while len(self.stream_out) >= self.filter_length:
            self.stream_out.popleft()

    def test(self, point1, point2):
        diff = self.last_diff
        if diff > (self.mean_in + self.mean_out) / 2:
            self.stream_in[-1] = self.mean_out - self.match_factor
            self.update_means()
        else:
            self.stream_out[-1] = self.mean_in + self.match_factor
            self.update_means()
        if np.abs(diff - self.mean_in) <= self.change_threshold:
            self.gesture_out = not self.defaults_to
            self.gesture_in = not self.gesture_out
        if np.abs(diff - self.mean_out) <= self.change_threshold:
            self.gesture_out = self.defaults_to
            self.gesture_in = not self.gesture_out



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
    global hist, hull
    key = cv2.waitKey(100)
    while key != ord('q'):
        frame_hist = cap.read()[1]
        frame_hist = cv2.flip(frame_hist, 1)
        working_frame = frame_hist[310:370, 380:420]
        image_hsv = cv2.cvtColor(working_frame, cv2.COLOR_BGR2HSV)
        hist_temp = cv2.calcHist([image_hsv], [0, 1], None, [256, 256], (0, 256, 0, 256))
        if hist is not None:
            contours, *_ = calc_motion_frame_contours(frame_hist, [237,393,327,443])
            if len(contours) > 0:
                hull = cv2.convexHull(contours[0])
                cv2.drawContours(frame_hist, [hull], -1, (255, 0, 0), 1)
        frame_hist[80:336,0:256] = cv2.cvtColor(hist_temp,cv2.COLOR_GRAY2BGR) * 255
        frame_hist[0:80, :] = (255,255,255)
        cv2.rectangle(frame_hist, (377, 307), (423, 373), color=(255, 255, 255), thickness=2)
        cv2.rectangle(frame_hist, (327, 237), (443, 393), color=(0, 0, 0), thickness=1)
        frame_hist = cv2.putText(frame_hist, "Place hand in the white box. Then", (0, 10), cv2.FONT_HERSHEY_PLAIN,fontScale=1, thickness=1, color=(0, 0, 0))
        frame_hist = cv2.putText(frame_hist, "press 'r' to record current picture, ", (0, 20), cv2.FONT_HERSHEY_PLAIN,fontScale=1, thickness=1, color=(0, 0, 0))
        frame_hist = cv2.putText(frame_hist, "press 'q' to quit or" , (0, 30), cv2.FONT_HERSHEY_PLAIN,fontScale=1, thickness=1, color=(0, 0, 0))
        frame_hist = cv2.putText(frame_hist, "press 'c' to clear the histogram's history.", (0, 40), cv2.FONT_HERSHEY_PLAIN,fontScale=1, thickness=1, color=(0, 0, 0))
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
    cv2.destroyWindow('color_signature')
    cv2.destroyWindow('color_signature_hist')


def draw_motion_frame(frame_motion_subframe, image_dimensions):
    frame_motion_subframe = cv2.line(frame_motion_subframe, (image_dimensions[2], image_dimensions[0]),
                                     (image_dimensions[3], image_dimensions[0]), (255, 40, 255), thickness=2)
    frame_motion_subframe = cv2.line(frame_motion_subframe, (image_dimensions[2], image_dimensions[1]),
                                     (image_dimensions[3], image_dimensions[1]), (255, 40, 255), thickness=2)
    frame_motion_subframe = cv2.line(frame_motion_subframe, (image_dimensions[2], image_dimensions[0]),
                                     (image_dimensions[2], image_dimensions[1]), (255, 40, 255), thickness=2)
    frame_motion_subframe = cv2.line(frame_motion_subframe, (image_dimensions[3], image_dimensions[0]),
                                     (image_dimensions[3], image_dimensions[1]), (255, 40, 255), thickness=2)
    return frame_motion_subframe


def save_image(thresh,name='', video=False):
    export_frame = thresh.copy()
    print("saving to " + control_parameters['image_label'])
    dirname = ('.\\images\\' + str(control_parameters['image_label']) + '\\')
    try:
        os.mkdir((dirname))
    except:
        pass
    if video:
        out.write(export_frame)
    else:
        cv2.imwrite(dirname + name +str(time.time()) + '.png',export_frame)


def calc_motion_contours(stream):
    kernel = np.ones((video_parameters['kernel_size_full_frame'], video_parameters['kernel_size_full_frame']), np.uint8)
    frame_xor = cv2.cvtColor(stream[-2], cv2.COLOR_BGR2GRAY)
    frame_xor = cv2.absdiff(frame_xor, cv2.cvtColor(stream[-1], cv2.COLOR_BGR2GRAY))
    frame_eroded = cv2.erode(frame_xor, kernel, iterations=1)
    frame_morphred = cv2.morphologyEx(frame_eroded, cv2.MORPH_OPEN, kernel)
    ret, thresh = cv2.threshold(frame_morphred, video_parameters['threshold_full_frame'], 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, frame_xor, frame_eroded, frame_morphred, thresh


def calc_motion_frame_contours(frame, image_dimensions):
    working_frame = frame[image_dimensions[0]:image_dimensions[1], image_dimensions[2]:image_dimensions[3]]
    # Get pointer to video frames from primary device
    image_hsv = cv2.cvtColor(working_frame.copy(), cv2.COLOR_BGR2HSV)
    if hist is None:
        min_HSV = np.array([0, 58, 30])
        max_HSV = np.array([33, 255, 255])
        skinRegionHSV = cv2.inRange(image_hsv, min_HSV, max_HSV)
    else:
        skinRegionHSV = cv2.calcBackProject([image_hsv], [0, 1], hist, (0, 256, 0, 256), 255)
        hist_temp = cv2.calcHist([image_hsv], [0, 1], None, [256, 256], (0, 256, 0, 256))
    kernel = np.ones((video_parameters['kernel_size_focus_frame'], video_parameters['kernel_size_focus_frame']), np.uint8)
    frame_dilated = cv2.dilate(skinRegionHSV, kernel, iterations=1)
    ret, thresh = cv2.threshold(frame_dilated, video_parameters['threshold_focus_frame'], 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(image_dimensions[2], image_dimensions[0]))
    # filters the countours. only consider contours with more points than parameters['min_contour_length']
    hist_matching_score = np.sum(np.abs(hist - hist_temp))
    contours = [con for con in contours if len(con) > filter_parameters['min_contour_length']]
    return contours, hist_matching_score, working_frame, frame_dilated


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
    left_points = np.array([segment_moving['left'][0]] + 4*[segment_stationary['left'][0]])
    right_points = np.array([segment_moving['right'][0]] + 4*[segment_stationary['right'][0]])
    up_points = np.array([segment_moving['top'][1]] + 4*[segment_stationary['top'][1]])
    down_points = np.array([segment_moving['down'][1]] + 4*[segment_stationary['down'][1]])
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
        if time.time() - control_parameters['pressed'] > 0.2:
            control_parameters['pressed'] = time.time()
            mouse.click(Button.left,1)
    if mb_left_double_click:
        if time.time() - control_parameters['pressed'] > 0.2:
            control_parameters['pressed'] = time.time()
            mouse.click(Button.left,2)
    if mb_right:
        if time.time() - control_parameters['pressed'] > 0.2:
            control_parameters['pressed'] = time.time()
            mouse.click(Button.right, 1)


thumb_gesture_controller = CentroidFilter('thumb',match_factor=5, change_threshold = 30, defaults_to=False, filter_length=5)
pinki_gesture_controller = CentroidFilter('pinki',match_factor=5, change_threshold = 30, defaults_to=False, filter_length=5)
top_gesture_controller = CentroidFilter('palm', change_threshold=20, match_factor=40)
def run(cap = cap):
    control_parameters['run'] = True
    image_dimensions = [0, 480, 0, 640]
    area_temp = 640*480//2
    centeroid_pt = (0,0)
    stream = deque()
    temp_hull = None
    line_stream = deque(np.array([(0, 0), (0, 0), (0, 0), (0, 0)]))
    shape_matching_index_stream = deque(np.array([0.05] * filter_parameters['gesture_detection_filter_size']))
    positions = {}
    segment_moving = {}
    segment_stationary = {}

    [ok, frame] = cap.read()
    stream.append(frame.copy())
    while ok and control_parameters['run']:
        top.update()

        #process keys
        key = cv2.waitKey(video_parameters['wait_time'])
        if key == ord('s'):
            control_parameters['save_current_image'] = True
        if key == ord('c'):
            control_parameters['record_video'] = True
        elif key == ord('x'):
            control_parameters['record_video'] = False

        # clear queues
        while len(stream) >= 2:
            stream.popleft()
        while len(shape_matching_index_stream) >= filter_parameters['gesture_detection_filter_size']:
            shape_matching_index_stream.popleft()
        while len(line_stream) >= filter_parameters['line_stream_length']:
            line_stream.popleft()

        # grab new frame
        positions['proc_time'] = time.time()
        [ok, frame] = cap.read()
        frame = cv2.flip(frame, 1)
        stream.append(frame.copy())

        # calculate countours and segments - motion
        contours, frame_xor, frame_eroded, frame_morphred, thresh = calc_motion_contours(stream.copy())
        if len(contours) > 0:
            merged_contours_motion = []
            for i, con in enumerate(contours):
                merged_contours_motion.extend(con)
            hull_full_frame = cv2.convexHull(np.array(merged_contours_motion))
            calc_segments([hull_full_frame], segment_moving)
        if control_parameters['save_current_image']:
            save_image(stream[-2], name='stream0')
            save_image(stream[-1], name='stream1')
            save_image(frame_xor,name='frame_xor')
            save_image(frame_eroded,name='frame_eroded')
            save_image(frame_morphred, name='frame_morphed')
            save_image(thresh, name='frame_thresh')

        # calculate countours and segments - sub frame
        contours_sub_Frame, hist_matching_score , sub_frame, thresh = calc_motion_frame_contours(frame.copy(), image_dimensions)
        if len(contours_sub_Frame) > 0:
            # find centroid from the top contour
            if hull is not None:
                matching_indicies = []
                hulls = []
                positions['hull_matching_index'] = np.mean(np.array(shape_matching_index_stream))
                while len(hulls) == 0 and positions['hull_matching_index'] <= filter_parameters['shape_matching_threshold']:
                    for con in contours_sub_Frame:
                        temp_hull = cv2.convexHull(np.array(con))
                        temp_matching_score = cv2.matchShapes(temp_hull, hull, cv2.CONTOURS_MATCH_I3, 0.0)
                        if temp_matching_score <= positions['hull_matching_index']:
                            hulls.append(temp_hull)
                            matching_indicies.append(temp_matching_score)
                    if len(hulls) == 0:
                        positions['hull_matching_index'] += 0.05
                if len(hulls) > 0:
                    hull = hulls[np.argmin(np.array(matching_indicies))]
                    positions['hull_matching_index'] = np.min(np.array(matching_indicies))
                    shape_matching_index_stream.append(np.min(np.array(matching_indicies)))
                else:
                    def calc_hull_and_shape(contours,prev_hull):
                        if len(contours) > 1:
                            merged_contours = []
                            [merged_contours.extend(con) for con in contours]
                        hull = cv2.convexHull(np.array(merged_contours))
                        if prev_hull is not None:
                            matching_score = cv2.matchShapes(prev_hull, hull, cv2.CONTOURS_MATCH_I3, 0.0)
                            return  hull, matching_score

                    e, k = 0, len(contours_sub_Frame)
                    positions['hull_matching_index'] = np.mean(np.array(shape_matching_index_stream))
                    temp_hulls = [np.array([]), np.array([]), np.array([])]
                    temp_matching_score = np.array([2.0, 2.0, 2.0])
                    while e < k - 1 and np.min(temp_matching_score) > positions['hull_matching_index']:
                        temp_hulls[0], temp_matching_score[0] = calc_hull_and_shape(contours_sub_Frame[e:k], hull)
                        if temp_matching_score[0] > positions['hull_matching_index']:
                            temp_hulls[1], temp_matching_score[1] = calc_hull_and_shape(contours_sub_Frame[e + 1:k], hull)
                            temp_hulls[1], temp_matching_score[1] = calc_hull_and_shape(contours_sub_Frame[e:k - 1], hull)
                            if temp_matching_score[1] > temp_matching_score[2]:
                                k -= 1
                            else:
                                e += 1
                    if np.min(temp_matching_score) <= positions['hull_matching_index']:
                        print("hull found by searching")
                        hull = temp_hulls[(np.argmin(temp_matching_score))]
                        positions['hull_matching_index'] = np.min(temp_matching_score)
                        shape_matching_index_stream.append(np.min(temp_matching_score))
                    else:
                        positions['hull_matching_index'] = filter_parameters['shape_matching_threshold']
            if hull is None or positions.get('hull_matching_index') == filter_parameters['shape_matching_threshold']:
                merged_contours = []
                [merged_contours.extend(con) for con in contours_sub_Frame]
                hull = cv2.convexHull(np.array(merged_contours))
                temp_hull = hull.copy()

            # calculate centroid
            centeroid_pt = centroid(hull)
            line_stream.append(centeroid_pt)
            calc_segments([hull], segment_stationary)
            thumb_gesture_controller.release_points()
            pinki_gesture_controller.release_points()
            top_gesture_controller.release_points()
            thumb_gesture_controller.add(segment_stationary['left'][0], centeroid_pt[0])
            pinki_gesture_controller.add(centeroid_pt[0], segment_stationary['right'][0])
            top_gesture_controller.add(segment_stationary['top'][1], segment_stationary['down'][1])

            # calculate the new sub frame
            ratio, area_temp, temp_dims = calc_motion_frame_dimensions(segment_stationary, segment_stationary)
            if area_temp >= filter_parameters['min_area']:
                image_dimensions = temp_dims
        elif len(contours) > 0 and gesture_control.get('top'):
            ratio, area_temp, temp_dims = calc_motion_frame_dimensions(segment_moving, segment_moving)
            if area_temp >= filter_parameters['min_area']:
                image_dimensions = temp_dims
        else:
            shape_matching_index_stream.append(shape_matching_index_stream[-1])
            if np.abs(np.mean(shape_matching_index_stream) - shape_matching_index_stream[-1]) < 0.0005:
                image_dimensions = [120, 360, 160, 480]


        if control_parameters['save_current_image']:
            save_image(sub_frame, name='sub_frame')
            save_image(thresh, name='thresh_2')
            control_parameters['save_current_image'] = False

        # This position values are used to adjust the mouse cursor coordinates.
        # Every iteration the mouse will move 'disp' (for displacement) pixels multiplied by the speed
        positions['proc_time'] = time.time() - positions['proc_time'] + 0.001*video_parameters['wait_time']
        if positions.get('top_mean'):
            positions['prev_top_mean'] = positions['top_mean']
            positions['top_mean'] = tuple(np.mean(np.array(line_stream), axis=0).astype(int))
            positions['disp'] = np.array(positions['top_mean']) - np.array(positions['prev_top_mean'])
        else:
            positions['top_mean'] = tuple(np.mean(np.array(line_stream), axis=0).astype(int))
            positions['disp'] = np.array(positions['top_mean']) - np.array((0,0))
        positions['motion_speed'] = np.abs(positions['disp'] / positions['proc_time'])


        if control_parameters['control']:
            if top_gesture_controller.gesture_in:
                control_by_method(positions, DispOnly=True)
                if pinki_gesture_controller.gesture_out and thumb_gesture_controller.gesture_out:
                    control_by_method(positions, mb_right=True)
                elif thumb_gesture_controller.gesture_out:
                    control_by_method(positions, mb_left=True)
            if top_gesture_controller.gesture_out:
                control_by_method(positions, SquareSpeed=True)

        frame_conts = cv2.drawContours(frame.copy(), contours_sub_Frame, -1, (0, 255, 0), 1)
        if hull is not None:
            frame_conts = cv2.drawContours(frame_conts, [hull], -1, (255, 0, 0), 1)
            frame_conts = cv2.drawContours(frame_conts, [temp_hull], -1, (255, 255, 128), 1)
            frame_conts = cv2.drawMarker(frame_conts, centeroid_pt,(255,255,255),cv2.MARKER_CROSS, thickness=2 )
            frame_conts = cv2.drawMarker(frame_conts, tuple(segment_stationary['top']), (255, 255, 255), cv2.MARKER_CROSS, thickness=2)
            frame_conts = cv2.drawMarker(frame_conts, positions['top_mean'], (0,0,0), cv2.MARKER_CROSS, thickness=2)
            if control_parameters['record_video']:
                save_image(frame_conts,'video',True)
        frame_motion_subframe = draw_motion_frame(frame_conts, image_dimensions)
        cv2.imshow('motion sub frame', frame_motion_subframe)

        color = (255, 255, 255)
        gesture_status_frame = cv2.putText(np.zeros((400, 300)), str(top_gesture_controller), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, color)
        gesture_status_frame = cv2.putText(gesture_status_frame, str(pinki_gesture_controller), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, color)
        gesture_status_frame = cv2.putText(gesture_status_frame, str(thumb_gesture_controller), (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, color)
        gesture_status_frame = cv2.putText(gesture_status_frame, str("hist_matching_Score: " + str(hist_matching_score)), (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, color)
        cv2.imshow('gesture status', gesture_status_frame)


video_parameters['kernel_size_full_frame'] = 3
video_parameters['kernel_size_focus_frame'] = 3
video_parameters['wait_time'] = 1
video_parameters['threshold_focus_frame'] = 50
video_parameters['threshold_full_frame'] = 20
video_parameters['focus_frame_margin'] = 90

filter_parameters['line_stream_length'] = 3
filter_parameters['min_contour_length'] = 90
filter_parameters['min_area'] = 20000
filter_parameters['gesture_detection_filter_size'] = 10
filter_parameters['shape_matching_threshold'] = 0.1

control_parameters['pressed'] = time.time()
control_parameters['control_speed'] = 0.002
control_parameters['run'] = True
control_parameters['control'] = False
control_parameters['record_video'] = False
control_parameters['save_current_image'] = False
control_parameters['image_label'] = ''

gesture_control['thumb'] = False
gesture_control['pinki'] = False
gesture_control['top_finger'] = False
gesture_control['top_fist'] = False

scales = {}
keys = ['kernel_size_full_frame', 'kernel_size_focus_frame', 'wait_time', 'threshold_full_frame', 'threshold_focus_frame', 'focus_frame_margin', 'shape_matching_threshold', 'min_contour_length', 'control_speed']
for i, key in enumerate(keys):
    scales[key] = tk.Scale(top, label = key, length=300, from_ = 1, to = 10, tickinterval = 1, orient='horizontal')
    if i >= (len(keys)//2):
        scales[key].grid(row=i - (len(keys)//2), column=4, columnspan=4, rowspan=1)
    else:
        scales[key].grid(row=i, column=0, columnspan=4, rowspan=1)


scales['kernel_size_full_frame'].config( command = lambda x : set_parameter(int(x),'kernel_size_full_frame',video_parameters))
scales['kernel_size_focus_frame'].config( command = lambda x : set_parameter(int(x),'kernel_size_focus_frame',video_parameters))
scales['wait_time'].config(to= 20,              command = lambda x : set_parameter(int(x),'wait_time',video_parameters,fraction=0.2))
scales['threshold_full_frame'].config(to= 25 , command = lambda x : set_parameter(int(x),'threshold_full_frame',video_parameters,fraction=0.1))
scales['threshold_focus_frame'].config(to= 25 , command = lambda x : set_parameter(int(x),'threshold_focus_frame',video_parameters,fraction=0.1))
scales['focus_frame_margin'].config(to= 20,                 command = lambda x : set_parameter(int(x),'focus_frame_margin',video_parameters,fraction=0.2))
scales['shape_matching_threshold'].config(to= 20 , command = lambda x : set_parameter(int(x),'shape_matching_threshold',filter_parameters,fraction=40))
scales['min_contour_length'].config(to= 25,     command = lambda x : set_parameter(int(x),'min_contour_length',filter_parameters,fraction=0.1))
scales['control_speed'].config(                         command = lambda x : set_parameter(int(x),'control_speed',control_parameters,fraction=1000))

scales['kernel_size_full_frame'].set(3)
scales['kernel_size_focus_frame'].set(1)
scales['wait_time'].set(1)
scales['threshold_full_frame'].set(2)
scales['threshold_focus_frame'].set(7)
scales['focus_frame_margin'].set(9)
scales['shape_matching_threshold'].set(6)
scales['min_contour_length'].set(6)
scales['control_speed'].set(6)

#tutorial_button = tk.Button(top, text="Tutorial",command=tutorial)
#tutorial_button.grid(row=0, column= 8)
get_color_signature_button = tk.Button(top, text="Get Color Signature",command=get_color_signature)
get_color_signature_button.grid(row=1, column= 8)
control_checkbox = tk.Checkbutton(top, text="control", command= lambda: set_parameter(not control_parameters['control'],'control',control_parameters))
control_checkbox.grid(row=2, column=8)
quit_button = tk.Button(top, text="Quit",command=quit)
quit_button.grid(row=4, column= 8)

get_color_signature()
run()

