import requests

def download_gfw_effort(token, output_path="data/sample_gfw.csv",
                        start_date="2023-07-01", end_date="2023-07-07",
                        bbox=[[79.0, 8.0], [81.0, 8.0], [81.0, 11.0], [79.0, 11.0], [79.0, 8.0]]):
    """
    Downloads GFW fishing effort CSV for a given date range and bounding box.

    Args:
        token (str): Your API token from GFW.
        output_path (str): Path to save the downloaded CSV.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        bbox (list): List of [lon, lat] coordinates forming a closed polygon.
    """
    url = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "date_range": f"{start_date},{end_date}",
        "datasets": [
            {
                "dataset": "public-global-fishing-effort",
                "version": "latest"
            }
        ],
        "spatial-resolution": "LOW",
        "temporal-resolution": "DAILY",
        "spatial-aggregation": False,
        "region": {
            "geometry": {
                "type": "Polygon",
                "coordinates": [bbox]
            }
        },
        "format": "CSV"
    }

    print(f"📡 Requesting GFW data from {start_date} to {end_date}...")

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Download complete: {output_path}")
    else:
        print("❌ Download failed.")
        print(f"Status code: {response.status_code}")
        print(response.text)


# Example usage
if __name__ == "__main__":
    your_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImtpZEtleSJ9.eyJkYXRhIjp7Im5hbWUiOiJORU1PIiwidXNlcklkIjo0ODI1OCwiYXBwbGljYXRpb25OYW1lIjoiTkVNTyIsImlkIjoyOTkzLCJ0eXBlIjoidXNlci1hcHBsaWNhdGlvbiJ9LCJpYXQiOjE3NTI2NTMwNjcsImV4cCI6MjA2ODAxMzA2NywiYXVkIjoiZ2Z3IiwiaXNzIjoiZ2Z3In0.HOmAMF2n4iSE_U1OzS9EaXJjfZVTAkCQlYxXLtuL9hGaQPLrBndbeN1ER6q_mjc-oF7OHdD7nq2xdMDZDXQrJ9hpe4Omy8Wb4XCKSyf0HLpjy9cp5YOAAjdENtX16jMRAEFUh4JxOpO3H0tz9ADM2Sa3RxI7KiYbothMy4YogsPUVwbUrCcpkSGiZiC6NP4_6d0GyF6idsisCQDZSGaeje52aCqzu0Z__O9tCS_ZeDIXWlsVyJZcJ3ii1lDlWkomS2M1cUiyNt95_TahEZOW2vU6dGLhpwTF7Rfv9VCjIedvk2X89ouaaxoEvLEhvrdfv4UO5f1N4FwY82gDjqpx6vZBbxvTWEeONvC33PYSmfjGc2zio3F9XrInPPA6nV6ojTEqqe-DznloCH0pFLAqRvxntN-q7dZOK3lilR-ZRC3AG8RtliJrynSEQYPETf1dWphpM2NvOUJeMUygan7-TAyKEo13fg3RwKTqDRnqQ2xGWaTwdD5ZpeOVnIZ2sWyf"
    download_gfw_effort(your_token)
