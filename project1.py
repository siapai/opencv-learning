import cv2
import numpy as np

frame_width = 640
frame_height = 480
cap = cv2.VideoCapture(1)
cap.set(3, frame_width)
cap.set(4, frame_height)
cap.set(10, 150)

myColors = [[20, 160, 150, 97, 255, 255]]

myColorsValues = [[0, 255, 192]]

myPoints = []  # x, y, colorId


def find_color(img_input, my_colors_input, my_colors_value):
    img_hsv = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV)
    color_index = 0
    points = []
    for color in my_colors_input:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(img_hsv, lower, upper)
        x, y = get_contours(mask)
        cv2.circle(img_result, (x, y), 10, my_colors_value[color_index], cv2.FILLED)
        if x != 0 and y != 0:
            points.append([x, y, color_index])
        color_index += 1
    return points


def get_contours(img_input):
    contours, hierarchy = cv2.findContours(img_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            # cv2.drawContours(img_result, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2, y


def draw_on_canvas(my_points, my_color_values):
    for my_point in my_points:
        cv2.circle(img_result, (my_point[0], my_point[1]), 10, my_color_values[point[2]], cv2.FILLED)


while True:
    success, img = cap.read()
    img_result = img.copy()
    new_points = find_color(img, myColors, myColorsValues)
    if len(new_points) != 0:
        for point in new_points:
            myPoints.append(point)
    if len(myPoints) != 0:
        draw_on_canvas(myPoints, myColorsValues)
    cv2.imshow("Original", img_result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
