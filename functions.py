import cv2
import tkinter as tk
import numpy as np

def get_color_signature(cap,hist):
    key = cv2.waitKey(100)
    while key != ord('q'):
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
    return hist


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


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

def area(dimensions):
    return np.abs(dimensions[0] - dimensions[1])*np.abs(dimensions[2] - dimensions[3])
def is_crossing_contour(contours, line):
    for con in contours:
        pass

def draw_line_object_tracker(frame,line,mean=True):
    if mean:
        start = tuple(np.mean(line.points_a, axis=0).astype(int))
        end = tuple(np.mean(line.points_a, axis=0).astype(int))
        cv2.drawMarker(frame, start, (0, 0, 255), cv2.MARKER_CROSS, thickness=2)
        cv2.drawMarker(frame, end, (0, 0, 255), cv2.MARKER_CROSS, thickness=2)
        cv2.line(frame,start,end,(0,0,255),thickness=1)
    else:
        cv2.drawMarker(frame, tuple(line.point_a), (0, 0, 255), cv2.MARKER_CROSS, thickness=2)
        cv2.drawMarker(frame, tuple(line.point_b), (0, 0, 255), cv2.MARKER_CROSS, thickness=2)
        cv2.line(frame, tuple(line.point_a), tuple(line.point_b), (0, 0, 255), thickness=1)

def try_merging(frame, dimensions, contours):
    if(area(dimensions) < 400):
        print('merge')
    else:
        if is_crossing_contour(dimensions,contours):
            pass