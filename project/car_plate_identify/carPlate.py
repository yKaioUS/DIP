import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from CNN import CNN
import sys

def recognize_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "图片读取失败，请检查路径或文件是否存在！"
    img = cv2.resize(img, (500,400))
    # cv2.imshow("img", img)
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow("blured", blured)
    
    # 边缘检测
    y = cv2.Sobel(blured, cv2.CV_16S, 1, 0)
    absY = cv2.convertScaleAbs(y)
    # cv2.imshow("absY", absY)
    
    # 二值化
    ret, thresh = cv2.threshold(absY, 75, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)
    
    # 开操作分割
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("open", open)
    
    # 闭操作填充
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 15))
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("closed", closed)
    
    kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
    
    erode = cv2.erode(close, kernel_x)
    dilate = cv2.dilate(erode, kernel_x)
    dilate = cv2.erode(dilate, kernel_y)
    erode = cv2.dilate(dilate, kernel_y)
    
    kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    erode = cv2.morphologyEx(erode, cv2.MORPH_ERODE, kernel_e)
    kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    dilate = cv2.morphologyEx(erode, cv2.MORPH_DILATE, kernel_d)
    # cv2.imshow("dilate", dilate)
    
    counters, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_copy, counters, -1, (255,0,255), 2)
    # cv2.imshow("img_copy1", img_copy)
    # cv2.waitKey(0)
    
    plate_img = None
    for counter in counters:
        rect = cv2.boundingRect(counter)
        if(rect[2] > rect[3]*2) and (rect[2] < rect[3]*6) and rect[2]*rect[3] > 5000 and rect[2]*rect[3] < 40000:
            plate_img = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            # cv2.imshow("plate1", plate_img)
            break
    
    if plate_img is None:
        print("未检测到车牌！")
        exit()
    else:
        plate_img = cv2.resize(plate_img, (800,400))
        cv2.imwrite(os.path.join(os.path.dirname(__file__), "..", "result/car_plate.jpg"), plate_img)
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        # 二值化
        ret, plate_img = cv2.threshold(plate_img, 127, 255, cv2.THRESH_BINARY)
        plate_img = plate_img[20:380,:]
        # cv2.imshow("plate", plate_img)
        cv2.imwrite(os.path.join(os.path.dirname(__file__), "..", "result/plate.jpg"), plate_img)
    
        # 按指定区间切分图片
        split_points = [(20,127), (127,233), (266,366), (366,476), (476,573), (573,680),(680,780)]
        for i, (x1, x2) in enumerate(split_points):
            char_img = plate_img[:, x1:x2]
            char_img = cv2.erode(char_img, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            char_img = cv2.resize(char_img, (28, 28))
            # cv2.imshow(f"char_{i}", char_img)
            cur_path = f"plate_char_{i}.jpg"
            cv2.imwrite(os.path.join(os.path.dirname(__file__), "..", "result", cur_path), char_img)
    
    # 加载类别名
    class_names = sorted(os.listdir(os.path.join(os.path.dirname(__file__), "./cnn_char_train")))
    num_classes = len(class_names)
    
    # 加载模型参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=num_classes).to(device)  # 加载模型并将其移动到GPU上，如果可用的话
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), './char_cnn.pth'), map_location=device))
    model.eval()
    
    # 定义transform
    transform = transforms.Compose([transforms.ToTensor()])
    
    # 预测
    result = []
    for i in range(7):
        cur_path = f"plate_char_{i}.jpg"
        char_img = Image.open(os.path.join(os.path.join(os.path.dirname(__file__), '..', 'result'), cur_path)).convert('L')
        char_img = transform(char_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(char_img)
            pred = output.argmax(dim=1).item()
            result.append(class_names[pred])
    # 最后return结果字符串
    return 'result: ' + ' '.join(result)

if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(recognize_plate(image_path))
    else:
        print("请传入图片路径")
    cv2.waitKey(0)
    '''
    for i in range(1,11):
        print(recognize_plate(f'./test/{i}.jpg'))
        cv2.waitKey(0)
    '''