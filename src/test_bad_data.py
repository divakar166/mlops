import requests

BASE_URL = "http://localhost:8000"

print("Testing API with various bad inputs...\n")

# Test 1: Negative amount
print("Test 1: Negative amount")
response = requests.post(f"{BASE_URL}/predict", json={
    "amount": -500.0,        # Negative amount - impossible!
    "hour": 14,
    "day_of_week": 3,
    "merchant_category": "online"
})
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}\n")

# Test 2: Invalid hour
print("Test 2: Hour = 25 (should be 0-23)")
response = requests.post(f"{BASE_URL}/predict", json={
    "amount": 100.0,
    "hour": 25,              # Invalid hour!
    "day_of_week": 3,
    "merchant_category": "online"
})
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}\n")

# Test 3: Invalid day of week
print("Test 3: day_of_week = 10 (should be 0-6)")
response = requests.post(f"{BASE_URL}/predict", json={
    "amount": 100.0,
    "hour": 14,
    "day_of_week": 10,       # Invalid day!
    "merchant_category": "online"
})
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}\n")

# Test 4: Unknown merchant category
print("Test 4: Unknown merchant category")
response = requests.post(f"{BASE_URL}/predict", json={
    "amount": 100.0,
    "hour": 14,
    "day_of_week": 3,
    "merchant_category": "unknown_category"  # Not in training data!
})
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}\n")

# Test 5: All bad at once
print("Test 5: Everything wrong")
response = requests.post(f"{BASE_URL}/predict", json={
    "amount": -1000.0,
    "hour": 99,
    "day_of_week": 15,
    "merchant_category": "totally_fake"
})
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}\n")

print("Observation: The API happily accepts ALL garbage and returns predictions!")
print("This is dangerous - bad data leads to bad predictions with no warning.")