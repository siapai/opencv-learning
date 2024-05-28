import cv2
import numpy as np

frame_width = 640
frame_height = 480
cap = cv2.VideoCapture(1)
cap.set(3, frame_width)
cap.set(4, frame_height)
cap.set(10, 150)

imageWidth = 480
imageHeight = 640


def pre_processing(img_input):
    img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 200, 200)
    kernel = np.ones((5, 5))
    img_dial = cv2.dilate(img_canny, kernel, iterations=2)
    img_threshold = cv2.erode(img_dial, kernel, iterations=1)
    return img_threshold


def get_contours(img_input):
    biggest = np.array([])
    max_area = 0
    contours, hierarchy = cv2.findContours(img_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            obj_corner = len(approx)
            if area > max_area and obj_corner == 4:
                biggest = approx
                max_area = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest


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


def reorder(points_input):
    points_input = points_input.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)
    add = points_input.sum(1)

    new_points[0] = points_input[np.argmin(add)]
    new_points[3] = points_input[np.argmax(add)]
    diff = np.diff(points_input, axis=1)
    new_points[1] = points_input[np.argmin(diff)]
    new_points[2] = points_input[np.argmax(diff)]
    return new_points


def get_warp(img_input, biggest_input):
    biggest_input = reorder(biggest_input)
    pts1 = np.float32(biggest_input)
    pts2 = np.float32([[0, 0], [imageWidth, 0], [0, imageHeight], [imageWidth, imageHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_output = cv2.warpPerspective(img_input, matrix, (imageWidth, imageHeight))
    img_cropped = img_output[20:img_output.shape[0]-20, 20:img_output.shape[1]-20]
    img_cropped = cv2.resize(img_cropped, (imageWidth, imageHeight))
    return img_cropped


while True:
    # success, img = cap.read()
    img = cv2.imread("Resources/document.jpg")
    cv2.resize(img, (imageWidth, imageHeight))
    imgContour = img.copy()
    imgThreshold = pre_processing(img)
    imgBiggest = get_contours(imgThreshold)
    if imgBiggest.size != 0:
        imgWrapped = get_warp(img, imgBiggest)
        imageArray = ([img, imgThreshold], [imgContour, imgWrapped])
    else:
        imageArray = ([img, imgThreshold], [img, img])

    stackedImages = stack_images(0.6, imageArray)
    cv2.imshow("Result", stackedImages)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
