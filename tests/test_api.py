import pytest
from app import app
from io import BytesIO

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_upscale_post(client):
    """Проверка отправки изображения"""
    data = {
        'image': (BytesIO(b"fake_image_data"), 'test.png')
    }
    response = client.post('/upscale', data=data, content_type='multipart/form-data')
    assert response.status_code == 201
    assert 'task_id' in response.json

def test_get_task_status(client):
    """Проверка получения статуса (задача не существует)"""
    # Для несуществующего ID Celery обычно возвращает PENDING
    response = client.get('/tasks/invalid_id')
    assert response.status_code == 200
    assert response.json['status'] == 'PENDING'

def test_processed_file_not_found(client):
    """Проверка получения файла, который еще не готов"""
    response = client.get('/processed/invalid_id')
    assert response.status_code == 404