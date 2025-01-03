import cv2
import numpy as np
import onnxruntime
from flask import Flask, request, jsonify, render_template
from flask_mail import Mail, Message
import os
import shutil

# Flask 初始化
app = Flask(__name__)

# 設置 Email 配置
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # 使用 Gmail SMTP 服務
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')  # 環境變數
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')  # 環境變數
mail = Mail(app)

# 載入 ONNX 模型
model_path = "yolov8.onnx"  # 確保模型文件在項目根目錄
session = onnxruntime.InferenceSession(model_path)

def preprocess_image(image_path):
    """預處理輸入圖像"""
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (640, 640))  # YOLO 模型預設輸入大小
    input_blob = np.expand_dims(resized_img.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
    return img, input_blob

def detect_puzzle(image_path):
    """使用 ONNX 模型檢測拼圖"""
    original_img, input_blob = preprocess_image(image_path)
    outputs = session.run(None, {"images": input_blob})  # 推論
    detections = outputs[0][0]  # 提取檢測結果

    # 繪製檢測框
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        if confidence < 0.5:  # 過濾低信心的檢測結果
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_img, f"{int(class_id)}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 保存處理後的圖像
    output_path = "temp/output_image.png"
    cv2.imwrite(output_path, original_img)
    return output_path

@app.route('/')
def home():
    """顯示前端頁面"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_images():
    """API 接口處理圖像"""
    # 確認是否有上傳圖像
    if 'image' not in request.files:
        return jsonify({"error": "Please upload an image"}), 400

    # 儲存圖像
    image = request.files['image']
    os.makedirs("temp", exist_ok=True)
    image_path = f"temp/{image.filename}"
    image.save(image_path)

    # 使用 ONNX 模型檢測
    output_path = detect_puzzle(image_path)

    # 如果用戶提供 Email，發送結果
    user_email = request.form.get('email')
    if user_email:
        send_email_with_attachment(user_email, output_path)

    # 清理臨時文件
    shutil.rmtree("temp", ignore_errors=True)

    return jsonify({"message": "Image processed successfully", "output": output_path})

def send_email_with_attachment(email, attachment_path):
    """發送包含附件的 Email"""
    msg = Message("Puzzle Analysis Result",
                  sender=app.config['MAIL_USERNAME'],
                  recipients=[email])
    msg.body = "Attached is the analysis result of your uploaded puzzle."
    with open(attachment_path, "rb") as f:
        msg.attach("output_image.png", "image/png", f.read())
    mail.send(msg)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
