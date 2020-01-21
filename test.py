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