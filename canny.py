#-*- coding:utf-8 -*-
import cv2
import numpy as np

name = "madori01"

# 入力画像を読み込み
img = cv2.imread("./original/" + name + ".jpg")

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 方法2(OpenCVで実装)
dst = cv2.Canny(gray, 600, 500)

# 結果を出力
cv2.imwrite("./results/" + name + ".jpg", dst)
