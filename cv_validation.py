#file for cv validation when setup
import torch
import cv2
import numpy as np
from kalmanfilter import KalmanFilter
#load model
model = torch.hub.load('.', 'custom', path='Trained_model/cv4control_v5n.onnx', source='local')

kf = KalmanFilter()
cap = cv2.VideoCapture('2743.mp4')

while(True):
    ret, frame = cap.read()
    predict_x = 0
    predict_y = 0
    img = frame
    try:
        img = model(frame)
        x0 = img.pandas().xyxy[0].to_dict('records')[0].get("xmin")
        y0 = img.pandas().xyxy[0].to_dict('records')[0].get("ymin")
        x1 = img.pandas().xyxy[0].to_dict('records')[0].get("xmax")
        y1 = img.pandas().xyxy[0].to_dict('records')[0].get("ymax")
        x_cen,y_cen = int((x0+x1)/2)-320,240-int((y0+y1)/2)
        print(x_cen,y_cen)
        predict_x, predict_y = kf.predict(x_cen,y_cen)
    except Exception as e: 
        print(e)
    out_img = np.squeeze(img.render())
    cv2.circle(out_img, (predict_x,predict_y), 20, (0, 0, 255), -1)
    cv2.imshow('YOLO', out_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()