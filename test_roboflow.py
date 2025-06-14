import requests
import base64

# Test different API endpoints and formats
API_KEY = "U9N0SYyfFJ7R5uJn7kYX"

print("=== Testing API Key Validity ===")
try:
    # Try to get account info
    account_url = "https://api.roboflow.com/account"
    response = requests.get(account_url, params={"api_key": API_KEY})
    print(f"Account status: {response.status_code}")
    print(f"Account response: {response.text[:300]}")
except Exception as e:
    print(f"Account error: {e}")

print("\n=== Testing Available Models ===")
try:
    # Try to get projects/models
    projects_url = "https://api.roboflow.com/projects"
    response = requests.get(projects_url, params={"api_key": API_KEY})
    print(f"Projects status: {response.status_code}")
    print(f"Projects response: {response.text[:500]}")
except Exception as e:
    print(f"Projects error: {e}")

# Test with different model formats
print("\n=== Testing Different Model Formats ===")
model_variations = [
    "skin-disease-ak",
    "skin_disease_ak",
    "skin-disease",
    "skin-disease-detection",
    "skin-disease-classification"
]

for model_name in model_variations:
    try:
        detect_url = f"https://detect.roboflow.com/{model_name}/1"
        response = requests.get(detect_url, params={"api_key": API_KEY})
        print(f"Model {model_name}: {response.status_code}")
        if response.status_code != 404:
            print(f"  Response: {response.text[:100]}")
    except Exception as e:
        print(f"Model {model_name}: Error - {e}")

# Test 1: Check if model exists
print("=== Testing Model Availability ===")
try:
    # Try to get model info
    model_url = f"https://api.roboflow.com/skin_disease_ak/1"
    response = requests.get(model_url, params={"api_key": API_KEY})
    print(f"Model info status: {response.status_code}")
    print(f"Model info response: {response.text[:200]}")
except Exception as e:
    print(f"Model info error: {e}")

# Test 2: Try detect endpoint with GET
print("\n=== Testing Detect Endpoint (GET) ===")
try:
    detect_url = f"https://detect.roboflow.com/skin_disease_ak/1"
    response = requests.get(detect_url, params={"api_key": API_KEY})
    print(f"Detect GET status: {response.status_code}")
    print(f"Detect GET response: {response.text[:200]}")
except Exception as e:
    print(f"Detect GET error: {e}")

# Test 3: Try serverless endpoint
print("\n=== Testing Serverless Endpoint ===")
try:
    serverless_url = f"https://serverless.roboflow.com/inference/skin_disease_ak/1"
    response = requests.get(serverless_url, params={"api_key": API_KEY})
    print(f"Serverless status: {response.status_code}")
    print(f"Serverless response: {response.text[:200]}")
except Exception as e:
    print(f"Serverless error: {e}")

# Test 4: Check workspace
print("\n=== Testing Workspace ===")
try:
    workspace_url = "https://api.roboflow.com/workspace"
    response = requests.get(workspace_url, params={"api_key": API_KEY})
    print(f"Workspace status: {response.status_code}")
    print(f"Workspace response: {response.text[:200]}")
except Exception as e:
    print(f"Workspace error: {e}") 