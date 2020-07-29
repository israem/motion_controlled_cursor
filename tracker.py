from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

class tracker:
    def __init__(self, threshhold,
                 flag_method = 'mean',
                 name = 'tracker',
                 length = 30,
                 def_flag = False,
                 cross_tracker_obj = False
                 ):
        self.length = length
        self.data = deque(np.array([0] * length))
        self.name = name
        self.flag = def_flag
        self.threshhold = threshhold
        self.flag_method = flag_method
        self.graph = None
        if cross_tracker_obj:
            self.cross_tracker_obj =  tracker(length = self.length,
                                              name = ('high_' + self.name),
                                              threshhold = self.threshhold,
                                              flag_method = self.flag_method,
                                              def_flag = self.flag
                                              )
        else:
            self.cross_tracker_obj = None


    def add(self,value):
        while (len(self.data) >= self.length):
            self.data.popleft()
        self.data.append(value)
        if self.cross_tracker_obj is not None:
            self.cross_tracker_obj.add(value)

    def set_length(self,length):
        self.length = length
        while (len(self.data) > self.length):
            self.data.popleft()
        if self.cross_tracker_obj is not None:
            self.cross_tracker_obj.set_length(length)

    def set_threshhold(self, threshhold):
        self.threshhold = threshhold
        if self.cross_tracker_obj is not None:
            self.cross_tracker_obj.set_threshhold(threshhold)


    def mean(self):
        return np.mean(np.array(self.data))

    def std(self):
        return np.std(np.array(self.data))

    def exec_flag_method(self):
        if self.flag_method == 'mean':
            return self.mean()
        if self.flag_method == 'std':
            return self.std()

    def cross_trackers(self, value,cross_value=None):
        if self.cross_tracker_obj is not None:
            self.cross_tracker_obj.cross_trackers(value,self.exec_flag_method())
            if value > (self.exec_flag_method() + self.cross_tracker_obj.exec_flag_method()) / 2:
                self.data[-1] = self.cross_tracker_obj.exec_flag_method() - 30
        else:
            if value <= (self.exec_flag_method() + cross_value) / 2:
                self.data[-1] = cross_value + 30


    def refresh_flag(self, value):
        if self.cross_tracker_obj is not None:
            if np.abs(value - self.mean()) <= self.threshhold:
                self.flag = True
                self.cross_tracker_obj.flag = False
            if np.abs(value - self.cross_tracker_obj.mean()) <= self.threshhold:
                self.flag = False
                self.cross_tracker_obj.flag = True
        else:
             self.flag = value > self.threshhold * self.exec_flag_method()
        return self.flag


    def graph_tracker(self):
        if self.graph is None:
            scaled_data = np.interp(self.data, (np.min(self.data), np.max(self.data)), (-300, 300)).astype(int)
            self.graph = graph(length=self.length, graph_mean=True)
            for i in scaled_data:
                self.graph.update(i)
        else:
            scaled_data = np.interp(self.data[-1], (np.min(self.data), np.max(self.data)), (-300, 300)).astype(int)
            self.graph.update(scaled_data)
        if self.cross_tracker_obj is not None:
            self.cross_tracker_obj.graph_tracker()





class graph:
    # def __init__(self, length = 200,y_range = 10000 , graph_mean = False,add_cross_graph = False):
    #     self.datay = deque(np.array([0]*length))
    #     self.datax = np.arange(0,(0.1*length),0.1)
    #     self.figure = plt.figure()
    #     self.figure.lines = plt.plot(
    #         self.datax,self.datay,'r-',
    #         self.datax,self.datay,'bs',
    #         self.datax, self.datay,'g^')
    #     self.figure.axes.append(plt.axis([0,0.1*length,-y_range,y_range]))
    #     self.graph_mean = graph_mean
    #     self.length = length
    # def update(self,value1):
    #     self.datay.popleft()
    #     self.datay.append(value1)
    #     self.figure.lines[0].set_ydata(self.datay)
    #     #self.figure.lines[0].axes.set_ylim(np.min(self.datay) - 20,np.max(self.datay) + 20)
    #     if self.graph_mean:
    #         self.update_mean()
    #         self.update_std()
    #     #self.figure.show()
    #     plt.pause(0.05)
    #     return self.line,

    def __init__(self, ax, maxt=2, dt=0.02):
        self.ax = ax
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.ydata = [0]
        self.xdata = [0]
        self.lines = plt.Line2D(self.tdata, self.ydata),plt.Line2D(self.tdata, self.xdata)
        self.ax.add_line(self.lines[0])
        self.ax.add_line(self.lines[1])
        self.ax.set_ylim(-10000, 10000)
        self.ax.set_xlim(0, self.maxt)


    def update(self, y):
        lastt = self.tdata[-1]
        if lastt > self.tdata[0] + self.maxt:  # reset the arrays
            self.tdata = [self.tdata[-1]]
            self.ydata = [self.ydata[-1]]
            self.xdata = [self.xdata[-1]]
            self.ax.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
            self.ax.figure.canvas.draw()

        t = self.tdata[-1] + self.dt
        self.tdata.append(t)
        self.ydata.append(y[1])
        self.xdata.append(y[0])
        self.lines[0].set_data(self.tdata, self.ydata)
        self.lines[1].set_data(self.tdata, self.xdata)
        return self.lines


    def update_mean(self):
        y = np.array([np.mean(self.datay)] * 3)
        self.figure.lines[1].set_xdata([0,10,20])
        self.figure.lines[1].set_ydata(y)
    def update_std(self):
        y = np.array([np.std(self.datay)] * 3)
        self.figure.lines[2].set_xdata([0, 10, 20])
        self.figure.lines[2].set_ydata(y)
    def add_mean(self):
        self.graph_mean = True



class line_object_tracker:
    def __init__(self,name,point_a = (0,0),point_b = (0,0),filter_size = 5):
        if not type(point_a) == np.ndarray:
            self.point_a = np.array(point_a)
        else:
            self.point_a = point_a
        if not type(point_b) == np.ndarray:
            self.point_b = np.array(point_b)
        else:
            self.point_b = point_b
        self.name = name
        self.points_a = deque()
        self.points_b = deque()
        self.filter_size = filter_size
    def set_points(self,point_a,point_b):
        if not type(point_a) == np.ndarray:
            self.point_a = np.array(point_a)
        else:
            self.point_a = point_a
        if not type(point_b) == np.ndarray:
            self.point_b = np.array(point_b)
        else:
            self.point_b = point_b
        self.update_filter()

    def update_filter(self):
        self.points_a.append(self.point_a)
        self.points_b.append(self.point_b)
        while len(self.points_a) > self.filter_size:
            self.points_a.popleft()
        while len(self.points_b) > self.filter_size:
            self.points_b.popleft()

    def draw(self):
        pass
    def length(self):
        return np.sqrt(np.sum((self.point_a - self.point_b) ** 2))
    def x_length(self):
        return np.abs(self.point_a[0] - self.point_a[0])
    def y_length(self):
        return np.abs(self.point_a[1] - self.point_a[1])
    def angle(self,line_object_tracker):
        # find crossing
        #return 2 angles
        pass

class shape_object_tracker:
    def __init__(self,name):
        self.lines = []
    def add_line(self,line):
        self.lines.append(line_object_tracker(name=line.name))
    def define_Shape(self):
        pass
    def set_angle_difference(self,angles):
        self.angles = angles
        pass
    def update_angles(self):

        pass





# np_res = np.array(res)
# fig, ax = plt.subplots()
# j = graph(ax,dt=0.01)
# ani = animation.FuncAnimation(fig, j.update, b, interval=10,
#                               blit=False)
# plt.show()
# filter_array_y = []
# filter_array_x = []
# sub_x  = 0
# sub_y  = 0
# new_arr = []
# for i,val in enumerate(a):
#     prev_disp = np.abs(a[i-1] -a[i-2])
#     curr_disp = np.abs(a[i] -a[i-1])
#     #print(prev_disp,curr_disp)
#     if (curr_disp[0] - prev_disp[0]) > 10:
#         filter_array_x.append(False)
#         sub_x = (curr_disp[0] - prev_disp[0])
#     else:
#         filter_array_x.append(True)
#         sub_x = 0
#     if (curr_disp[1] - prev_disp[1]) > 10:
#         filter_array_y.append(False)
#         sub_y = (curr_disp[1] - prev_disp[1])
#     else:
#         filter_array_y.append(True)
#         sub_y = 0
#     print(np.array([sub_x,sub_y]))
#     new_arr.append(np.array([sub_x,sub_y]))




def matrix_to_values(image):
    channel = [[] for i in np.arange(256)]
    chan_conv = []
    for i,val in enumerate(image):
        for j, val1 in enumerate(val):
            channel[int(val1)].append(np.array([i,j]))
    for val in channel:
        chan_conv.append(np.array(val))
    return np.array(chan_conv)


# hist = get_color_signature(cap,hist)
# image_hsv = cv2.cvtColor(working_frame.copy(), cv2.COLOR_BGR2HSV)
# if hist is None:
#     min_HSV = np.array([0, 58, 30])
#     max_HSV = np.array([33, 255, 255])
#     skinRegionHSV = cv2.inRange(image_hsv, min_HSV, max_HSV)
# else:
#     skinRegionHSV = cv2.calcBackProject([image_hsv], [0, 1], hist, (0, 256, 0, 256), 255)
#
# cv2.imshow('maskd',cv2.bitwise_and(frame, frame, mask = skinRegionHSV))
#
# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame,1)
#     if frame is None:
#         break
#     fgMask = backSub.apply(frame)
#     cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
#     cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
#                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#     cv2.imshow('Frame', frame)
#     cv2.imshow('FG Mask', fgMask)
#     cv2.imshow('maskd', cv2.bitwise_and(frame, frame, mask=fgMask))
#     keyboard = cv2.waitKey(30)
#     if keyboard == 'q' or keyboard == 27:
#         break
#
#
# working_frame = frame[image_dimensions[0]:image_dimensions[1], image_dimensions[2]:image_dimensions[3]]
# contours, hierarchy = cv2.findContours(working_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(image_dimensions[2], image_dimensions[0]))
# contours = [remove_close_points(con,video_parameters['smoothness_threshhold']) for con in contours if len(con) > filter_parameters['min_contour_length']]

cap = cv2.VideoCapture(0)
backSub = cv.createBackgroundSubtractorKNN()
ret,frame = cap.read()
# setup initial location of window
x, y, w, h = 300, 200, 100, 50 # simply hardcoded the values
track_window = (x, y, w, h)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
results = [np.array([240,320])]
while(1):
    ret, frame = cap.read()
    positions['proc_time'] = time.time()
    frame = cv2.flip(frame,1)
    if ret == True:
        fgMask = backSub.apply(frame)
        maskd = cv2.bitwise_and(frame, frame, mask=fgMask)
        hsv = cv2.cvtColor(maskd, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0,1],hist,[0,256,0,256],255)
        # apply camshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        results.append(np.array(ret[0]))
        positions['proc_time'] = time.time() - positions['proc_time'] + 0.001 * 1
        dist = np.sqrt(np.sum((np.int32(results[-1]) - np.int32(results[-2])) ** 2))
        dist_elem = results[-1] - results[-2]
        if dist <= 200:
            positions['disp'] = dist_elem
            positions['motion_speed'] = np.abs(positions['disp'] / positions['proc_time'])
            control_by_method(positions, SquareSpeed=True, mouse_action='move')
        else:
            print("skipping")
        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        cv2.imshow('img2',img2)
        cv2.imshow('dst', dst)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    else:
        break
        


def draw_results(results,waittime=30):
    frame = np.zeros((480, 640, 3))
    prev_point = np.array([480//2,640//2])
    for i,val in enumerate(results):
        #print([np.array([np.int32(val),np.int32(results[i-1])])])
        dist = np.sqrt(np.sum((np.int32(val) - np.int32(results[i - 1])) ** 2))
        dist_elem = val - results[i - 1]
        print(dist)
        color = (255,255,255)
        if dist > 20:
            color = (0,0,0)
        else:
            frame = cv2.polylines(frame, [np.array([prev_point,np.int32(prev_point + dist_elem) ])], True,color,1)
            prev_point = np.int32(prev_point + dist_elem)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(waittime) & 0xff

        
        
        
        
        
        
        
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (50,50),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(dst, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

