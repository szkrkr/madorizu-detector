# import
import cv2
import numpy as np
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# define the Fixed
MODEL_NAME = 'inference_graph'
FILE_NAME = 'madori01'
ORIGIN_IMAGE_NAME = './original/' + FILE_NAME + '.jpg'
IMAGE_SIZE = 0

# 画像の大きさ取得
def getImageSize(img):
    height, width, channels = img.shape
    return height * width

# remove small item
def removeSmallItem(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    (thresh, im_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bw = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)[1]
    return img_bw

#find all your connected components 
def findAllConnectedComponents(img_bw):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_bw, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 150  
    _img = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            _img[output == i + 1] = 255
    return _img

# dilate to keep wall
def dilation(_img):
    kernel = np.ones((6, 6), np.uint8)
    dilated_img = cv2.dilate(_img, kernel)
    return dilated_img

# find walls
def find_walls(origin, dst): 
    # 輪郭を抽出
    contours, hierarchy= cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    dst = origin
    for i, contour in enumerate(contours):
        # 小さな領域の場合は間引く
        area = cv2.contourArea(contour)
        if area < 500:
            continue
        # 画像全体を占める領域は除外する
        if IMAGE_SIZE * 0.99 < area:
            continue
        
        # 外接矩形を取得
        x,y,w,h = cv2.boundingRect(contour)
        dst = cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
    return dst

# find walls 2nd. it is divided from find rooms functin.
def find_walls_2(origin, img):
    # corners_threshold=0.1,
    # assert 0 <= corners_threshold <= 1
    # Remove noise left from door removal

    threshold = 10
    img[img < threshold] = 0
    img[img > threshold] = 255
    cv2.imwrite('./_.jpg', img)
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = cv2.drawContours(origin, contours, -1, (0, 0, 255, 255), 2, cv2.LINE_AA)
    return contours, result


# find rooms
# ref: https://stackoverflow.com/questions/54274610/separate-rooms-in-a-floor-plan-using-opencv
def find_rooms(img, noise_removal_threshold=25, corners_threshold=0.1,room_closing_max_length=100, gap_in_wall_threshold=500):
    """
    :param img: grey scale image of rooms, already eroded and doors removed etc.
    :param noise_removal_threshold: Minimal area of blobs to be kept.
    :param corners_threshold: Threshold to allow corners. Higher removes more of the house.
    :param room_closing_max_length: Maximum line length to add to close off open doors.
    :param gap_in_wall_threshold: Minimum number of pixels to identify component as room instead of hole in the wall.
    :return: rooms: list of numpy arrays containing boolean masks for each detected room
    colored_house: A colored version of the input image, where each room has a random color.
    """
    assert 0 <= corners_threshold <= 1
    # Remove noise left from door removal

    threshold = 10
    img[img < threshold] = 0
    img[img > threshold] = 255
    cv2.imwrite('./_.jpg', img)
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)
    for contour in contours:
        area = cv2.contourArea(contour)
        # print(area)
        if area > noise_removal_threshold:
            cv2.fillPoly(mask, [contour], 255)

    img = ~mask

    # Detect corners (you can play with the parameters here)
    dst = cv2.cornerHarris(img ,2,3,0.04)
    dst = cv2.dilate(dst,None)
    corners = dst > corners_threshold * dst.max()

    # Draw lines to close the rooms off by adding a line between corners on the same x or y coordinate
    # This gets some false positives.
    # You could try to disallow drawing through other existing lines for example.
    for y,row in enumerate(corners):
        x_same_y = np.argwhere(row)
        for x1, x2 in zip(x_same_y[:-1], x_same_y[1:]):
            if x2[0] - x1[0] < room_closing_max_length:
                color = 0
                cv2.line(img, (x1[0], y), (x2[0], y), color, 1)

    for x,col in enumerate(corners.T):
        y_same_x = np.argwhere(col)
        for y1, y2 in zip(y_same_x[:-1], y_same_x[1:]):
            if y2[0] - y1[0] < room_closing_max_length:
                color = 0
                cv2.line(img, (x, y1[0]), (x, y2[0]), color, 1)


    # Mark the outside of the house as black
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, [biggest_contour], 255)
    img[mask == 0] = 0

    # Find the connected components in the house
    ret, labels = cv2.connectedComponents(img)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    unique = np.unique(labels)
    rooms = []
    for label in unique:
        component = labels == label
        if img[component].sum() == 0 or np.count_nonzero(component) < gap_in_wall_threshold:
            color = 0
        else:
            rooms.append(component)
            color = np.random.randint(0, 255, size=3)
        img[component] = color

    return rooms, img


img = cv2.imread(ORIGIN_IMAGE_NAME, cv2.IMREAD_COLOR)
IMAGE_SIZE = getImageSize(img)

dst = removeSmallItem(img)
cv2.imwrite('./results/' + FILE_NAME + '-detector-01-smallitemremoved.jpg', dst)
# dst = findAllConnectedComponents(_)
# cv2.imwrite('./results/' + FILE_NAME + '-detector-02-allcomponentsfound.jpg', dst)
dst = dilation(dst)
cv2.imwrite('./results/' + FILE_NAME + '-detector-03-dilated.jpg', dst)
# rooms, colored_house = find_rooms(img.copy())
# cv2.imwrite('./results/' + FILE_NAME + '-detector-04-colored_house.jpg', colored_house)

# walls = find_walls(img, dst)
# cv2.imwrite('./results/' + FILE_NAME + '-detector-10-walls.jpg', walls)

contours, walls2 = find_walls_2(img, dst)
cv2.imwrite('./results/' + FILE_NAME + '-detector-11-walls_2.jpg', walls2)

# for contour in contours:
#     print(contour)
# result = contours.to_list() # nested lists with same data, indices


array = np.array(contours)
numpyData = {'result': array }
file_path = "./path.json" ## your path variable
# json.dump(numpyData, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
with open("./json/" + FILE_NAME + ".json", "w") as write_file:
    json.dump(numpyData, write_file, cls=NumpyArrayEncoder)

# json.dumps(numpyData)

_, rooms  = find_rooms(dst)
cv2.imwrite('./results/' + FILE_NAME + '-detector-12-rooms.jpg', rooms)