import os
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from celery import Celery, result
from cv2 import dnn_superres

app = Flask(__name__)

# Настройки лимитов и форматов
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Лимит 16 МБ
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Настройка Celery
app.config['CELERY_BROKER_URL'] = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6373/0')
app.config['CELERY_RESULT_BACKEND'] = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6373/0')

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

SCALER = None


def get_scaler():
    global SCALER
    if SCALER is None:
        SCALER = dnn_superres.DnnSuperResImpl_create()
        SCALER.readModel('EDSR_x2.pb')
        SCALER.setModel("edsr", 2)
    return SCALER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@celery.task(bind=True)
def upscale_task(self, image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Рекомендация: проверка на битый файл
    if image is None:
        return None

    scaler = get_scaler()
    result_image = scaler.upsample(image)

    _, buffer = cv2.imencode('.png', result_image)
    return buffer.tobytes()


# --- Роуты Flask ---

@app.post('/upscale')
def post_upscale():
    if 'image' not in request.files:
        return jsonify({"error": "No image field"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Исправление №2: Валидация расширения
    if not allowed_file(file.filename):
        return jsonify({"error": f"Allowed types are: {ALLOWED_EXTENSIONS}"}), 400

    image_bytes = file.read()
    task = upscale_task.delay(image_bytes)

    # Исправление №1: Статус 202 (Accepted) вместо 201
    return jsonify({"task_id": task.id}), 202


@app.get('/tasks/<task_id>')
def get_status(task_id):
    task_result = result.AsyncResult(task_id, app=celery)
    response = {"status": task_result.status}

    if task_result.status == 'SUCCESS':
        if task_result.result is None:  # Если cv2 не смог прочитать файл
            return jsonify({"status": "FAILURE", "error": "Invalid image data"}), 400
        response["file_url"] = f"/processed/{task_id}"

    return jsonify(response)


@app.get('/processed/<task_id>')
def get_file(task_id):
    task_result = result.AsyncResult(task_id, app=celery)
    if task_result.status != 'SUCCESS' or task_result.result is None:
        return jsonify({"error": "Task not finished or failed"}), 404

    image_io = io.BytesIO(task_result.result)
    return send_file(image_io, mimetype='image/png', as_attachment=True, download_name=f"{task_id}.png")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)