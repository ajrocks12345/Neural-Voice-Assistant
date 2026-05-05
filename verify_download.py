import requests
import os

def test_download():
    # Test for an existing user in logs
    name = "akssa"
    age = "24"
    url = f"http://127.0.0.1:5000/download-history?name={name}&age={age}"
    
    print(f"Requesting download for {name}_{age}...")
    try:
        r = requests.get(url, timeout=5)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print(f"Content-Disposition: {r.headers.get('Content-Disposition')}")
            print(f"Content-Type: {r.headers.get('Content-Type')}")
            # Save it temporarily to verify content
            with open("test_download.csv", "wb") as f:
                f.write(r.content)
            print(f"File saved to test_download.csv (Size: {len(r.content)} bytes)")
        else:
            print(f"Error: {r.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_download()
