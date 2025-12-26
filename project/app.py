from flask import Flask, request, jsonify, send_from_directory
import os
import subprocess
from basic_picture_process.basic_process import process_image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
RESULT_FOLDER = 'result/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('.', 'main.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'msg': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'msg': 'No selected file'})
    filename = file.filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    return jsonify({'success': True, 'url': '/uploads/' + filename, 'filename': filename})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/result/<filename>')
def result_file(filename):
    try:
        return send_from_directory('result', filename)
    except Exception as e:
        print(f"访问文件错误: {str(e)}")
        return jsonify({'error': '文件不存在'}), 404

@app.route('/recognize', methods=['POST'])
def recognize_plate():
    data = request.get_json()
    filename = data.get('filename')
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result = subprocess.check_output(['python', 'car_plate_identify/carPlate.py', img_path], encoding='utf-8', errors='ignore')
    return jsonify({'result': result})

@app.route('/recognize_plus', methods=['POST'])
def recognize_plate_plus():
    data = request.get_json()
    filename = data.get('filename')
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result = subprocess.check_output(['python', 'car_plate_identify/carPlate_plus.py', img_path], encoding='utf-8', errors='ignore')
    return jsonify({'result': result})

@app.route('/cell_count', methods=['POST'])
def cell_count_api():
    data = request.get_json()
    filename = data.get('filename')
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result = subprocess.check_output(['python', 'cell_count/cell.py', img_path], encoding='utf-8', errors='ignore')
    # 提取数字
    try:
        count = int(result.strip().split('=')[-1])
    except Exception:
        count = None
    return jsonify({'count': count})

@app.route('/process', methods=['POST'])
def process_image_api():
    try:
        data = request.get_json()
        filename = data.get('filename')
        operation = data.get('operation')
        params = data.get('params', {})
        
        if not filename or not operation:
            return jsonify({
                'success': False,
                'msg': '缺少必要参数'
            })
        
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(img_path):
            return jsonify({
                'success': False,
                'msg': '图片文件不存在'
            })
        
        if 'operation' in params:
            params.pop('operation')
            
        result_path = process_image(img_path, operation, **params)
        
        if result_path and os.path.exists(result_path):
            return jsonify({
                'success': True,
                'url': '/' + result_path
            })
        else:
            return jsonify({
                'success': False,
                'msg': '图像处理失败'
            })
    except Exception as e:
        print(f"处理错误: {str(e)}")  # 添加错误日志
        return jsonify({
            'success': False,
            'msg': f'处理出错: {str(e)}'
        })

@app.route('/car.html')
def car_html():
    return send_from_directory('car_plate_identify', 'car.html')

@app.route('/car_plus.html')
def car_plus_html():
    return send_from_directory('car_plate_identify', 'car_plus.html')

@app.route('/cell.html')
def cell_html():
    return send_from_directory('cell_count', 'cell.html')

@app.route('/basic_process.html')
def basic_process_html():
    return send_from_directory('basic_picture_process', 'basic_process.html')

if __name__ == '__main__':
    app.run(debug=True)