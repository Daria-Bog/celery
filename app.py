import os
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from celery import Celery, result
from cv2 import dnn_superres

app = Flask(__name__)

# Настройка Celery
app.config['CELERY_BROKER_URL'] = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6373/0')
app.config['CELERY_RESULT_BACKEND'] = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6373/0')

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Глобальный объект для модели (загружается один раз при запуске воркера)
SCALER = None


def get_scaler():
    global SCALER
    if SCALER is None:
        SCALER = dnn_superres.DnnSuperResImpl_create()
        SCALER.readModel('EDSR_x2.pb')
        SCALER.setModel("edsr", 2)
    return SCALER


@celery.task(bind=True)
def upscale_task(self, image_bytes):
    # Превращаем байты обратно в картинку для OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    scaler = get_scaler()
    result_image = scaler.upsample(image)

    # Кодируем результат обратно в байты (PNG), чтобы не писать на диск
    _, buffer = cv2.imencode('.png', result_image)
    return buffer.tobytes()


# --- Роуты Flask ---

@app.post('/upscale')
def post_upscale():
    if 'image' not in request.files:
        return jsonify({"error": "No image field"}), 400

    file = request.files['image']
    image_bytes = file.read()

    task = upscale_task.delay(image_bytes)
    return jsonify({"task_id": task.id}), 201


@app.get('/tasks/<task_id>')
def get_status(task_id):
    task_result = result.AsyncResult(task_id, app=celery)
    response = {"status": task_result.status}

    if task_result.status == 'SUCCESS':
        # Ссылка на файл
        response["file_url"] = f"/processed/{task_id}"

    return jsonify(response)


@app.get('/processed/<task_id>')
def get_file(task_id):
    task_result = result.AsyncResult(task_id, app=celery)
    if task_result.status != 'SUCCESS':
        return jsonify({"error": "Task not finished"}), 404

    # Отдаем файл прямо из памяти
    image_io = io.BytesIO(task_result.result)
    return send_file(image_io, mimetype='image/png', as_attachment=True, download_name=f"{task_id}.png")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)