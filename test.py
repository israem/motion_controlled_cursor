images = []
key = cv2.waitKey(100)
while key != ord('a'):
    frame_hist = cap.read()[1]
    frame_hist = cv2.flip(frame_hist, 1)
    cv2.rectangle(frame_hist, (377, 307), (423, 373), color=(255, 255, 255), thickness=2)
    #show_images([frame_hist], ['color_signature'])
    cv2.imshow('color_signature', frame_hist)
    working_frame = frame_hist[310:370, 380:420]
    imageHSV = cv2.cvtColor(working_frame, cv2.COLOR_BGR2HSV)
    key= cv2.waitKey(20)
    if key == ord('r'):
        frame_hist = cap.read()[1]
        frame_hist = cv2.flip(frame_hist, 1)
        cv2.rectangle(frame_hist, (377, 307), (423, 373), color=(255, 255, 255), thickness=2)
        cv2.imshow('color_signature', frame_hist)
        print(frame_hist.shape)
        working_frame = frame_hist[310:370, 380:420]
        imageHSV = cv2.cvtColor(working_frame, cv2.COLOR_BGR2HSV)
        images.append(imageHSV)
        key = cv2.waitKey()

        # thumb_mean_stream = deque(np.array([0] * filter_parameters['gesture_detection_filter_size']))
        # pinki_mean_stream = deque(np.array([0] * filter_parameters['gesture_detection_filter_size']))
        # top_f_mean_stream = deque(np.array([0] * filter_parameters['gesture_detection_filter_size']))
        # top_fist_mean_stream = deque(np.array([0] * filter_parameters['gesture_detection_filter_size']))
        # while len(thumb_mean_stream) >= filter_parameters['gesture_detection_filter_size']:
        #     thumb_mean_stream.popleft()
        # while len(pinki_mean_stream) >= filter_parameters['gesture_detection_filter_size']:
        #     pinki_mean_stream.popleft()
        # while len(top_f_mean_stream) >= filter_parameters['gesture_detection_filter_size']:
        #     top_f_mean_stream.popleft()
        # while len(top_fist_mean_stream) >= filter_parameters['gesture_detection_filter_size']:
        #     top_fist_mean_stream.popleft()



        # Gesture control, the queues acting as filters to measure the mean distances over time.
        # So we can dinamically detect whether a significat motion was done if the current is much larger then the mean
        # since every new value continuously fed into the filter the mean value will adjust itself,
        # essentialy allowing the gesture to be performed only once over time.

        #thumb_mean_stream.append((np.array(centeroid_pt) - segment_stationary['left'])[0])
        #positions['thumb_mean'] = np.mean(np.array(thumb_mean_stream))
        #pinki_mean_stream.append((segment_stationary['right'] - np.array(centeroid_pt))[0])
        #positions['pinki_mean'] = np.mean(np.array(pinki_mean_stream))
        #top_f_mean_stream.append((np.array(segment_stationary['down']) - segment_stationary['top'])[1])
        #positions['top_f_mean'] = np.mean(np.array(top_f_mean_stream))
        #top_fist_mean_stream.append((np.array(segment_stationary['down']) - segment_stationary['top'])[1])
        #positions['top_fist_mean'] = np.mean(np.array(top_fist_mean_stream))
        # diff = np.abs((np.array(segment_stationary['down']) - segment_stationary['top'])[1])
        # if diff > (positions['top_fist_mean'] + positions['top_f_mean']) / 2:
        #     # top_fist_mean_stream.pop()
        #     top_fist_mean_stream[-1] = positions['top_f_mean'] - 30
        #     positions['top_fist_mean'] = np.mean(np.array(top_fist_mean_stream))
        # else:
        #     top_f_mean_stream[-1] = positions['top_fist_mean'] + 30
        #     positions['top_f_mean'] = np.mean(np.array(top_f_mean_stream))
        #
        # gesture_control['thumb'] = (np.array(centeroid_pt) - segment_stationary['left'])[0] > 1.5 * positions[
        #     'thumb_mean']
        # gesture_control['pinki'] = (segment_stationary['right'] - np.array(centeroid_pt))[0] > 1.7 * positions[
        #     'pinki_mean']
        # if np.abs(diff - positions['top_fist_mean']) <= 20:
        #     gesture_control['top_fist'] = True
        #     gesture_control['top_finger'] = False
        # if np.abs(diff - positions['top_f_mean']) <= 20 and not gesture_control['pinki'] and not gesture_control[
        #     'thumb']:
        #     gesture_control['top_finger'] = True
        #     gesture_control['top_fist'] = False
        # if control_parameters['control']:
        #     if gesture_control.get('top_fist'):
        #         control_by_method(positions, DispOnly=True)
        #         if gesture_control['pinki'] and gesture_control['thumb']:
        #             control_by_method(positions, mb_right=True)
        #         elif gesture_control['thumb']:
        #             control_by_method(positions, mb_left=True)
        #     if gesture_control['top_finger']:
        #         control_by_method(positions, SquareSpeed=True)
