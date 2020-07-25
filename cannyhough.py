#-*- coding:utf-8 -*-
import cv2
import numpy as np

name = "madori01"

# 入力画像を読み込み
img = cv2.imread("./original/" + name + ".jpg")

# グレースケール変換
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Second, process edge detection use Canny.
low_threshold = 50
high_threshold = 100
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# canny結果を出力
cv2.imwrite("./results/" + name + "-canny.jpg", edges)


# Then, use HoughLinesP to get the lines. You can adjust the parameters for better performance.
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 50  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 30  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

# Draw the lines on the  image
# img_white = cv2.imread("./original/" + name + "white.jpg")
lines_edges = cv2.addWeighted(img, 0.1, line_image, 1, 0)

# canny結果を出力
cv2.imwrite("./results/" + name + "-hough.jpg", lines_edges)