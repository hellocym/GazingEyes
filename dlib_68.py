# 对静态人脸图像文件进行68个特征点的标定

import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
from gaze_tracking import GazeTracking

# GazeTracker 瞳孔检测
gaze = GazeTracking()

# Dlib 检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("E2-Capsnet-master/shape_predictor_68_face_landmarks.dat")

# 读取图像文件
#img_rd = cv2.imread("E2-Capsnet-master/RAF/train/train_00089_aligned.jpg")
video_capture = cv2.VideoCapture(1)
while True:
    _, img_rd = video_capture.read()
    try:
        gaze.refresh(img_rd)
    except cv2.error:
        pass
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    cv2.namedWindow("image", 0)
    cv2.imshow("image", img_rd)
    #cv2.waitKey(0)
    # 人脸数
    faces = detector(img_gray, 0)

    # 待会要写的字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    hor_c = 0.5
    ver_c = 0.5
    canvas1 = np.zeros((1000, 1000, 3), np.uint8)
    canvas1.fill(255)
    # 标 68 个点
    if len(faces) != 0:
        # 检测到人脸
        canvas1 = np.zeros((1000, 1000, 3), np.uint8)
        canvas1.fill(255)
        for i in range(len(faces)):
            # 取特征点坐标
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[i]).parts()])
            for idx, point in enumerate(landmarks):
                # 68 点的坐标
                pos = (point[0, 0], point[0, 1])

                # 利用 cv2.circle 给每个特征点画一个圈，共 68 个
                cv2.circle(img_rd, pos, 2, color=(139, 0, 0))
                # 利用 cv2.putText 写数字 1-68
                #cv2.putText(img_rd, str(idx + 1), pos, font, 0.5, (187, 255, 255), 1, cv2.LINE_AA)
            #print(landmarks[37:43][0, 0])
            left_eye = [(min(landmarks[36:42, 0]).item(), min(landmarks[36:42, 1]).item()), (max(landmarks[36:42, 0]).item(), max(landmarks[36:42, 1]).item())]
            right_eye = [(min(landmarks[42:48, 0]).item(), min(landmarks[42:48, 1]).item()), (max(landmarks[42:48, 0]).item(), max(landmarks[42:48, 1]).item())]
            #print(right_eye)
            #print(left_eye)
            cv2.rectangle(
                img_rd,
                right_eye[0],
                right_eye[1],
                (255, 0, 0),  # color of rectangle
                2  # thickness of rectangle
            )
            cv2.rectangle(
                img_rd,
                left_eye[0],
                left_eye[1],
                (255, 0, 0),  # color of rectangle
                2  # thickness of rectangle
            )
            cv2.drawMarker(
                img_rd,
                gaze.pupil_left_coords(),
                (0, 0, 255),
                markerSize=10
            )
            cv2.drawMarker(
                img_rd,
                gaze.pupil_right_coords(),
                (0, 0, 255),
                markerSize=10
            )
            try:
                #print(right_eye)
                #print(gaze.pupil_right_coords())
                #print((right_eye[1][0] - right_eye[0][0]))
                right_pupil_horizonal_movement = (gaze.pupil_right_coords()[0] - right_eye[0][0]) / (right_eye[1][0] - right_eye[0][0])
                #print(right_pupil_horizonal_movement)
                left_pupil_horizonal_movement = (gaze.pupil_left_coords()[0] - left_eye[0][0]) / (left_eye[1][0] - left_eye[0][0])

                right_pupil_vertical_movement = (gaze.pupil_right_coords()[1] - right_eye[0][1]) / (right_eye[1][1] - right_eye[0][1])
                left_pupil_vertical_movement = (gaze.pupil_left_coords()[1] - left_eye[0][1]) / (left_eye[1][1] - left_eye[0][1])
                avg_horizonal_movement = (right_pupil_horizonal_movement + left_pupil_horizonal_movement) / 2
                avg_vertical_movement = (right_pupil_vertical_movement + left_pupil_vertical_movement) / 2

                hor_ratio = (avg_horizonal_movement - hor_l) / (hor_r - hor_l)
                ver_ratio = (avg_vertical_movement - ver_d) / (ver_u - ver_d)
                #print(min(max(int(hor_ratio), 0), 1) * 1000)
                #print((min(max(int(hor_ratio), 0), 1) * 1000, min(max(int(ver_ratio), 0), 1) * 1000))
                #print((min(max(hor_ratio, 0), 1), min(max(ver_ratio, 0), 1)))
                cv2.circle(canvas1, (min(max(int(hor_ratio), 0), 1) * 1000, min(max(int(ver_ratio), 0), 1) * 1000) , 50, (0, 0, 255), 50)
            except Exception:
                pass
            #py_ratio[i] = max(min((py - (fy + ey)) / eh * 2 - 0.8, 1), 0)
            #if py_ratio[i] < 0.5:
            #    py_ratio[i] = 0.5 - abs(0.5 - py_ratio[i]) * 10
            #else:
            #    py_ratio[i] = 0.5 + abs(0.5 - py_ratio[i]) * 10
            #print((min(landmarks[37:43, 0]).item(), min(landmarks[37:43, 1]).item()))
            #print((max(landmarks[37:43, 0]).item(), max(landmarks[37:43, 1]).item()))
            #print((min(landmarks[42:49, 0]).item(), min(landmarks[42:49, 1]).item()))
            #print((max(landmarks[42:49, 0]).item(), max(landmarks[42:49, 1]).item()))
            #print((max(landmarks[42:49][0, 0]), max(landmarks[42:49][0, 1]))
        cv2.putText(img_rd,"", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        # 没有检测到人脸
        cv2.putText(img_rd,"", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

    # 窗口显示
    # 参数取 0 可以拖动缩放窗口，为 1 不可以
    # cv2.namedWindow("image", 0)

    cv2.namedWindow("image", 0)
    img_rd = cv2.flip(img_rd, 1)
    cv2.imshow("image", img_rd)
    cv2.imshow("vp", cv2.flip(canvas1, 1))
    if cv2.waitKey(10) & 0xFF==ord('q'):
        hor_c = avg_horizonal_movement
        ver_c = avg_vertical_movement
        #(avg_horizonal_movement - hor_c) * 2 + 0.5
        #print((avg_horizonal_movement - hor_c) * 2 + 0.5, (avg_vertical_movement - ver_c) * 2 + 0.5)
    if cv2.waitKey(10) & 0xFF==ord('a'):
        hor_l = avg_horizonal_movement
        ver_u = avg_vertical_movement
    if cv2.waitKey(10) & 0xFF==ord('d'):
        hor_r = avg_horizonal_movement
        ver_d = avg_vertical_movement