from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import torch
import cv2
import numpy as np
import imutils
import time
from kalmanfilter import KalmanFilter

model = torch.hub.load('.','custom', path='D:\\CV_cuda\\yolov5\\runs\\train\\yolov7-tut312\\weights\\last.engine', source='local')
# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture('2743.mp4')

prev_frame_time = 0
new_frame_time = 0
kf = KalmanFilter()

#function for determine midpoint of any 2 points
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
    
#function for determine red/blue armor plate
#if is red armor plate return True, if it's blue armor return False
def is_red(crop):
    crop=cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and blue
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    lower_blue = np.array([100, 120, 70])
    upper_blue = np.array([130, 255, 255])

    # Create masks for each color range
    red_mask = cv2.inRange(crop, lower_red, upper_red)
    blue_mask = cv2.inRange(crop, lower_blue, upper_blue)

    # Count the number of pixels for each color
    red_pixels = cv2.countNonZero(red_mask)
    blue_pixels = cv2.countNonZero(blue_mask)

    # Determine the dominant color
    if red_pixels > blue_pixels:
        return True
    elif blue_pixels > red_pixels:
        return False
 
 
#Main loop for camera  
while(True):
    ret, frame = cap.read()
    img = frame
    predict_x = 0
    predict_y = 0
    
    try:
        img = model(frame)
        x0 = img.pandas().xyxy[0].to_dict('records')[0].get("xmin")
        y0 = img.pandas().xyxy[0].to_dict('records')[0].get("ymin")
        x1 = img.pandas().xyxy[0].to_dict('records')[0].get("xmax")
        y1 = img.pandas().xyxy[0].to_dict('records')[0].get("ymax")
        x_cen,y_cen = (x0+x1)/2,(y0+y1)/2
        crop = frame[int(y0):int(y1),int(x0):int(x1)]
        print(x_cen,y_cen)
        predict_x, predict_y = kf.predict(x_cen,y_cen)
  
    except:
        pass 
        
    #determine color of armor plate
    if(is_red(crop)):
        print("is red")
    else:
        print("is blue")
        
    #pre-process for 1st target armor plate  
    gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    ret, thresh= cv2.threshold(gray, 200,255, cv2.THRESH_BINARY)
    gray = cv2.GaussianBlur(thresh,(5,5),0)   
    edged = cv2.Canny(gray,30,100)
    edged = cv2.dilate(edged, None, iterations=1)
    #edged = cv2.erode(edged, None, iterations=1)
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    orig = crop.copy() 
    center = []
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 50 or cv2.contourArea(c) > 500:
            continue
        # compute the rotated bounding box of the contour
        print(cv2.contourArea(c))
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tlbrX, tlbrY) = midpoint(tl, br)
        center.append((tlbrX, tlbrY))
        # (blbrX, blbrY) = midpoint(bl, br)
        # # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        # (tlblX, tlblY) = midpoint(tl, bl)
        # (trbrX, trbrY) = midpoint(tr, br)
        # # draw the midpoints on the image
        # cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        # cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        # cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        # cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        # # draw lines between the midpoints
        # cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
            # (255, 0, 255), 2)
        # cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
            # (255, 0, 255), 2)
    	# # compute the Euclidean distance between the midpoints
        # dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        # dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # # if the pixels per metric has not been initialized, then
        # # compute it as the ratio of pixels to supplied metric
        # # (in this case, inches)
        # if pixelsPerMetric is None:
            # pixelsPerMetric = dB / 0.5  
        # # compute the size of the object
        # dimA = dA / pixelsPerMetric
        # dimB = dB / pixelsPerMetric
        # # draw the object sizes on the image
        # cv2.putText(orig, "{:.1f}in".format(dimA),
            # (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            # 0.65, (255, 255, 255), 2)
        # cv2.putText(orig, "{:.1f}in".format(dimB),
            # (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            # 0.65, (255, 255, 255), 2)
    try:
        d_horizon = dist.euclidean(center[0], center[1])
        print(d_horizon) 
    except:
        print('missing light bar')
    out_img = np.squeeze(img.render())
    cv2.circle(out_img, (predict_x,predict_y), 20, (0, 0, 255), -1)
    thresh = np.resize(gray, orig.shape)
    orig = np.vstack((orig, thresh))
    cv2.imshow('YOLO', orig)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()