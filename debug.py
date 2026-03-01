"""Quick debug script — run this to see exactly what the exchange returns."""

import requests

EXCHANGE_URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com/"  # <-- paste your URL
USERNAME = "Market Fakers"
PASSWORD = "marketfakers123"                 # <-- paste your password

url = f"{EXCHANGE_URL.rstrip('/')}/api/user/authenticate"
print(f"POST {url}")
print(f"Body: username={USERNAME}, password=***\n")

resp = requests.post(
    url,
    headers={"Content-Type": "application/json; charset=utf-8"},
    json={"username": USERNAME, "password": PASSWORD},
)

print(f"Status: {resp.status_code}")
print(f"\nResponse Headers:")
for k, v in resp.headers.items():
    print(f"  {k}: {v}")
print(f"\nResponse Body:")
print(resp.text[:500])