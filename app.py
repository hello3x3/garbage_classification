import os
from gradio_client import Client, file
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="templates")
client = Client("http://127.0.0.1:50000/")

# 设置允许上传的文件类型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 检查文件扩展名是否合法
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

# 文件上传接口
@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查是否有文件被上传
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    my_file = request.files['file']
    
    # 检查文件名是否为空
    if my_file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # 检查文件类型是否合法
    if not allowed_file(my_file.filename):
        return jsonify({'error': 'Invalid file type'})

    # 保存上传的文件到指定目录
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    my_file.save(os.path.join(upload_folder, my_file.filename))

    result = client.predict(
		img=file(os.path.join(upload_folder, my_file.filename)),
		api_name="/predict"
    )
    
    return result

if __name__ == '__main__':
    app.run(debug=True, port=6006, host="0.0.0.0")
