import cv2
import numpy as np
# from pyzbar.pyzbar import decode
# from colorsys import hsv_to_rgb

# def stackImage(scale, imageArray):
#     rows = len(imageArray)
#     cols = len(imageArray[0])
#     rowsAvailable = isinstance(imageArray[0],list)
#     width = imageArray[0][0].shape[1]
#     height = imageArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range(0,rows):
#             for y in range(0, cols):
#                 if imageArray[x][y].shape[:2] == imageArray[0][0].shape[:2]:
#                     imageArray[x][y] = cv2.resize(imageArray[x][y],(0,0), None, scale, scale)
#                 else:
#                     imageArray[x][y] = cv2.resize(imageArray[x][y],(imageArray[0][0].shape[1], imageArray[0][0].shape[0]), None, scale, scale)
#                 if len(imageArray[x][y].shape)==2:
#                     imageArray[x][y] = cv2.cvtColor(imageArray[x][y], cv2.COLOR_GRAY2BGR)

#         imageBlank = np.zeros((height, width, 3), np.unit8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imageArray[x])
#         ver = np.vstack(hor)

#     else:
#         for x in range(0, rows):
#             if imageArray[x].shape[:2] == imageArray[0].shape[:2]:
#                 imageArray[x] = cv2.resize(imageArray[x],(0,0), None, scale, scale)
#             else:
#                 imageArray[x] = cv2.resize(imageArray[x],(imageArray[0].shape[1], imageArray[0].shape[0]), None, scale, scale)
#             if len(imageArray[x].shape)==2:
#                     imageArray[x] = cv2.cvtColor(imageArray[x], cv2.COLOR_GRAY2BGR)
#         hor = np.vstack(imageArray)
#         ver = hor
#     return ver
def crop(img):
    width, height = 600,600;
    pts1 = np.float32([[72,72],[544,100],[43,427],[528,461]])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    output = cv2.warpPerspective(img, matrix, (width,height))
    return output

# def getQRvalue(image):
#     return decode(image)[0].data.decode()


def blackMask(image):
    bool = False   # Check wether the function detects the color or not
    mask = cv2.inRange(image, (0, 0, 0), (35, 55, 100))
    if np.sum(mask)>0:
        bool = True     # if black colour detected it set bool to True
    return bool,mask

def redMask(image):
    bool = False
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red_mask = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
    upper_red_mask = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
    ## Merge the mask and crop the red regions
    mask = cv2.bitwise_or(lower_red_mask, upper_red_mask )
    if np.sum(mask)>0:
        bool = True
    return bool,mask

def getContoursAndCenter(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for cont in contours:
        area = cv2.contourArea(cont)
        
        minArea = cv2.getTrackbarPos("Area","Parameters")
        if area> minArea:
            cv2.drawContours(imgContour, cont, -1, (255,0,255), 3)
            peri = cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, 0.02*peri, True)
            moments = cv2.moments(cont)
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            print(cx, "  ",cy)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.circle(imgContour, (cx, cy), 5, (0, 0, 255), -1)
            # cv2.putText(imgContour, "Centroid", (cx - 25, cy - 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 4)
            cv2.rectangle(imgContour, (x,y), (x+w, y+h), (0,255,0), 3)
            # rect = cv2.minAreaRect(cont)
            # (cx,cy),(w,h), angle = rect
            # cv2.circle(img, (int(cx), int(cy)),5,(0,0,255), -1)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.polylines(img, [box], True, (255,0,0), 2)
            # print(box)






def empty(a):
    pass



frameWidth = 600
frameHeight = 600
cap = cv2.VideoCapture(2)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",146,255,empty)
cv2.createTrackbar("Threshold2","Parameters",112,255,empty)
cv2.createTrackbar("Area","Parameters",5000, 30000, empty)


while True:
    success, img = cap.read()
    img = cv2.flip(img,0)
    img = cv2.flip(img,1)
    img = crop(img)
    imgContour = img.copy()
    # imgBlur = cv2.GaussianBlur(img, (7,7),1)
    # imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1","Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2","Parameters")
    # imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
    # imgStack = stackImage(0.8,([img, imgCanny]))
    # kernel = np.ones((5,5))
    # imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    isBlack, bMask = blackMask(img)
    getContoursAndCenter(bMask, imgContour)
    cv2.imshow("Result",imgContour)
    if isBlack:
        print("Yes")
    else:
        print("No")
    isRed, rMask = redMask(img)
    getContoursAndCenter(rMask, imgContour)
    cv2.imshow("Result",imgContour)
    if isRed:
        print("Yes")
    else:
        print("No")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break