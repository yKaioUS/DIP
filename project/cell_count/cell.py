import cv2
import numpy as np
import sys
import os

def cell_count(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (500, 500))
    img_copy = img.copy()

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("thresh", thresh)

    # 闭运算
    kernel = np.ones((6, 6), int)
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("close", close)

    # 腐蚀
    kernel = np.ones((15, 15), int)
    erode = cv2.erode(close, kernel)
    # cv2.imshow("erode", erode)

    # 开运算
    kernel = np.ones((15, 15), int)
    open_img = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("open_img", open_img)

    # 绘制边缘
    contours, hierarchy = cv2.findContours(open_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_copy = cv2.drawContours(img_copy, contours, -1, (0, 0, 255), 1)
    # cv2.imshow("img_copy", img_copy)
    count = len(contours)

    # 保存处理后图片
    # 统一保存到 result 目录
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'result')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 修改保存文件名格式
    save_path = os.path.join(save_dir, "cell_result.jpg")
    cv2.imwrite(save_path, img_copy)
    return count

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        count = cell_count(image_path)
        print(f"count={count}")
    else:
        print("请传入图片路径")
    cv2.waitKey(0)
    '''
    for i in range(4,5):
        print(cell_count(f'./test_cell/{i}.jpg'))
        cv2.waitKey(0)
    '''