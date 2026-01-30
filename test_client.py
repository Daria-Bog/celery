import requests
import time

URL = "http://localhost:5000"

def test_service():
    # 1. Отправка файла
    with open('lama_300px.png', 'rb') as f:
        resp = requests.post(f"{URL}/upscale", files={'image': f})
        task_id = resp.json()['task_id']
        print(f"Task created: {task_id}")

    # 2. Ожидание
    while True:
        resp = requests.get(f"{URL}/tasks/{task_id}")
        status = resp.json()['status']
        print(f"Status: {status}")
        if status == 'SUCCESS':
            break
        time.sleep(2)

    # 3. Скачивание
    file_url = f"{URL}/processed/{task_id}"
    img_data = requests.get(file_url).content
    with open('result.png', 'wb') as f:
        f.write(img_data)
    print("Result saved as result.png")

if __name__ == '__main__':
    test_service()