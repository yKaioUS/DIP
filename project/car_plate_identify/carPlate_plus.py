import cv2
import os
from openai import OpenAI
import base64
from dotenv import load_dotenv, find_dotenv
import sys
from unidecode import unidecode

def recognize_plate_plus(image_path):
    img = cv2.imread(image_path) # 读取图片
    if img is None:
        return "图片读取失败，请检查路径或文件是否存在！"
    img = cv2.resize(img, (500,400)) # 调整图片大小
    # cv2.imshow("img", img)
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度化
    blured = cv2.GaussianBlur(gray, (5, 5), 0) # 高斯模糊
    # cv2.imshow("blured", blured)
    
    # 边缘检测
    y = cv2.Sobel(blured, cv2.CV_16S, 1, 0)
    absY = cv2.convertScaleAbs(y)
    # cv2.imshow("absY", absY)
    
    # 二值化
    ret, thresh = cv2.threshold(absY, 75, 255, cv2.THRESH_BINARY) # 大于75设置为255白色
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
    
    counters, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 检测轮廓
    cv2.drawContours(img_copy, counters, -1, (255,0,255), 2) 
    #cv2.imshow("img_copy", img_copy)
    #cv2.waitKey(0)
    
    plate_img = None
    # 根据矩形大小提取车牌
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

        # 将图像转换为Base64编码
        _, buffer = cv2.imencode('.jpg', plate_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64

def upload_image_to_tongyi(img):
    _=load_dotenv(find_dotenv())
    # 导入 OpenAI API_KEY 
    qwen_api_key=os.environ['OPENAI_API_KEY']
    client = OpenAI(
        api_key=os.getenv(qwen_api_key),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {
                "role": "user",
                "content":[
                    {
                        "type": "text",
                        "text": "请识别这张图片中的车牌号码。只返回车牌号码，不返回其他任何信息。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{img}"}
                    }
                ]
            },
        ]
    )
    response = completion.choices[0].message.content
    return response


if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        # 识别并处理车牌图像，获取Base64编码
        img_base64 = recognize_plate_plus(image_path)
        
        if img_base64:
            # 发送到大模型进行识别
            plate_number = upload_image_to_tongyi(img_base64)
            plate_number = plate_number.replace(" ", "")
            plate_number = unidecode(plate_number)

            if plate_number:
                print(f"{plate_number}")
            else:
                print("车牌识别失败")
    else:
        print("请传入图片路径")
        cv2.waitKey(0)
    '''
    image_path = f'./test/{12}.jpg'
    result = upload_image_to_tongyi(recognize_plate_plus(image_path))
    result = result.replace(" ", "")
    print(result)
    '''
