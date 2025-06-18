# Digital_Image_Processing

## About The Project
This project is used as a image processing system, including the following three functions:
* 1. basic image processing
* 2. license plate recognition(version: normal and plus)
* 3. cell counting


<!-- GETTING STARTED -->
## Getting Started
### Prerequisites
* 创建虚拟环境
  ```
    conda create -n opencv python=3.9
  ```

* 安装opencv依赖包
  ```
    pip install opencv-python==4.5.5.64
  ```

* 安装openai依赖包
  ```
    pip install openai
  ```

* 按照pytorch依赖包
  ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

* 按照其他依赖包
  ```
    pip install numpy==1.19.5
    pip install flask
    pip install unidecode
    pip install base64
    pip install python-dotenv
  ```

### Usage

1. 激活虚拟环境
   ```sh
   conda activate opencv
   ```
2. 运行项目文件
    ```sh
    python ./app.py
    ```
3. 打开浏览器进入http://127.0.0.1:5000/

### tips
* 车牌识别plus功能需要在./car_plate_indentify/.env中输入通义千问api_key

### reference
* 车牌提取的思路参考https://blog.csdn.net/great_yzl/article/details/119934992?spm=1001.2014.3001.5502
* 其余功能自研，杜绝抄袭！！
