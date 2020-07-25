import cv2 as cv
import numpy as np

def main():
    # ファイルを読み込み
    file_name = 'madori01'
    image_file = './original/' + file_name + '.jpg'
    src = cv.imread(image_file, cv.IMREAD_COLOR)
    # 画像の大きさ取得
    height, width, channels = src.shape
    image_size = height * width
    # グレースケール化
    img_gray = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    cv.imwrite('./results/' + file_name + '-imggray-debug.jpg', img_gray)
    # しきい値指定によるフィルタリング
    retval, dst = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY)
    cv.imwrite('./results/' + file_name + '-dst-debug.jpg', dst)
    ### HACK
    kernel = np.ones((5,5),np.uint8)
    erosion = cv.erode(dst,kernel,iterations = 1)
    cv.imwrite('./results/' + file_name + '-erosion.jpg', dst)
    ### 
    # 白黒の反転
    dst = cv.bitwise_not(dst)
    # 再度フィルタリング
    retval, dst = cv.threshold(dst, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imwrite('./results/' + file_name + '-dst2-debug.jpg', dst)
    # 輪郭を抽出
    # dst, contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) - これはversionによって使えない https://teratail.com/questions/194500
    contours, hierarchy= cv.findContours(dst,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
    # この時点での状態をデバッグ出力
    dst = cv.imread(image_file, cv.IMREAD_COLOR)
    dst = cv.drawContours(dst, contours, -1, (0, 0, 255, 255), 2, cv.LINE_AA)
    cv.imwrite('./results/' + file_name + '-findcontour-debug.jpg', dst)
    dst = cv.imread(image_file, cv.IMREAD_COLOR)
    for i, contour in enumerate(contours):
        # 小さな領域の場合は間引く
        area = cv.contourArea(contour)
        if area < 20000:
            continue
        # 画像全体を占める領域は除外する
        if image_size * 0.99 < area:
            continue
        
        # 外接矩形を取得
        x,y,w,h = cv.boundingRect(contour)
        dst = cv.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
    # 結果を保存
    cv.imwrite('./results/' + file_name + '-findcontour.jpg', dst)
    
if __name__ == '__main__':
    main()