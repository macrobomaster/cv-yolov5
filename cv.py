import torch
import cv2
import numpy as np
from kalmanfilter import KalmanFilter
import CvCmdApi
#load model
model = torch.hub.load('.', 'custom', path='Trained_model/cv4control_v5n.onnx', source='local')
#capture video from camera
cap = cv2.VideoCapture(1)
#cameara setup
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FOURCC, 0x32595559)
cap.set(cv2.CAP_PROP_FPS, 120)
#capture video from camera
# camera_id = "/dev/video0"
# cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# cap.set(cv2.CAP_PROP_FPS, 120)

cv2control = CvCmdApi.CvCmdHandler()
kf = KalmanFilter()
# cap = cv2.VideoCapture('')

while(True):
    ret, frame = cap.read()
    predict_x = 0
    predict_y = 0
    img = frame
    #control mode input
    oldflags = (False, False, False)
    try:
        img = model(frame)
        x0 = img.pandas().xyxy[0].to_dict('records')[0].get("xmin")
        y0 = img.pandas().xyxy[0].to_dict('records')[0].get("ymin")
        x1 = img.pandas().xyxy[0].to_dict('records')[0].get("xmax")
        y1 = img.pandas().xyxy[0].to_dict('records')[0].get("ymax")
        x_cen,y_cen = int((x0+x1)/2)-320,240-int((y0+y1)/2)
        print(x_cen,y_cen)
        predict_x, predict_y = kf.predict(x_cen,y_cen)
        cv2control.CvCmd_Heartbeat(gimbal_coordinate_x=predict_x,gimbal_coordinate_y=predict_y,chassis_speed_x=0,chassis_speed_y=0)
    except Exception as e: 
        print(e)
    out_img = np.squeeze(img.render())
    cv2.circle(out_img, (predict_x,predict_y), 20, (0, 0, 255), -1)
    cv2.imshow('YOLO', out_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()