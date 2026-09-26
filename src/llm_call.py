import requests

# Ваш URL без ключа внутри строки
URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent" 


# Настройка заголовков аутентификации
headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": "AQ.Ab8RN6JF8vsl026lI6zstpxrp64yHSIc2PIkEA5-wzuYWcWcGw",   # КЛЮЧ ЗДЕСЬ
}

data = {
    "contents": [
        {
            "parts": [{"text": "Скажи 'Привет', если ты меня слышишь."}]
        }
    ]
}

response = requests.post(URL, headers=headers, json=data)

if response.status_code == 200:
    print("Успешно! Ответ модели:")
    # print(response.json()['candidates']['content']['parts']['text'])
    print(response.json()['candidates'][0]['content']['parts'][0]['text'])
else:
    print(f"Ошибка {response.status_code}: {response.text}")
