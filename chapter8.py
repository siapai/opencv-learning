import cv2
import numpy as np


def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                          scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def get_contours(img1):
    contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            obj_corner = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if obj_corner == 3:
                object_type = "Tri"
            elif obj_corner == 4:
                aspect_ratio = w/float(h)
                if 0.95 < aspect_ratio < 1.05:
                    object_type = "Square"
                else:
                    object_type = "Rectangle"
            elif obj_corner > 4:
                object_type = "Circle"
            else:
                object_type = "None"

            cv2.rectangle(img_contour, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                img_contour,
                object_type,
                (x+(w//2)-10, y+(h//2)-10),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0, 0, 0),
                2
            )


path = "Resources/shapes.png"
img = cv2.imread(path)
img_contour = img.copy()

imgBlank = np.zeros_like(img)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)
get_contours(imgCanny)

img_stack = stack_images(0.68, ([img, imgGray, imgBlur], [imgCanny, img_contour, imgBlank]))
cv2.imshow("Results", img_stack)
cv2.waitKey(0)
