import requests
import time

def test_sentiment():
    url = "http://127.0.0.1:5000/predict"
    payload = {
        "user_name": "test_user",
        "user_age": "20",
        "message": "i am not feeling good",
        "history": [],
        "user_profile": {}
    }
    
    print("Sending request for 'i am not feeling good'...")
    start = time.time()
    try:
        r = requests.post(url, json=payload, timeout=5)
        print(f"Elapsed: {time.time() - start:.2f}s")
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.json().get('response')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_sentiment()
