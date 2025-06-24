import os
import requests
import json
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import io
from typing import List, Tuple, Optional

class SentinelHubFetcher:
    def __init__(self, client_id: str, client_secret: str):
        """
        Initialize the Sentinel Hub API client with robust image handling
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.token_expires = None
        self.base_url = "https://services.sentinel-hub.com"
        self.bbox_size = 0.02  # Default bounding box size in degrees (~2km)
        self.resolution = 20    # Changed to 20m resolution for better compatibility

    def get_oauth_token(self) -> str:
        """Get OAuth token for API authentication"""
        if self.token and self.token_expires and datetime.now() < self.token_expires:
            return self.token

        token_url = f"{self.base_url}/oauth/token"
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }

        try:
            response = requests.post(token_url, data=data)
            response.raise_for_status()
            token_data = response.json()
            self.token = token_data['access_token']
            self.token_expires = datetime.now() + timedelta(seconds=token_data['expires_in'] - 300)
            return self.token
        except Exception as e:
            print(f"Error getting OAuth token: {e}")
            return None

    def create_evalscript(self) -> str:
        """
        Simplified evalscript for better compatibility
        """
        return """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04", "B08"],
                output: { 
                    bands: 4,
                    sampleType: "UINT16"
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02, sample.B08];
        }
        """

    def fetch_image(self,
                    latitude: float,
                    longitude: float,
                    bbox_size: float = None,
                    start_date: str = None,
                    end_date: str = None,
                    resolution: int = None,
                    max_cloud_coverage: int = 30) -> Optional[np.ndarray]:
        """
        Fetch satellite image with robust error handling
        """
        if not self.get_oauth_token():
            print("Authentication failed")
            return None

        bbox_size = bbox_size if bbox_size is not None else self.bbox_size
        resolution = resolution if resolution is not None else self.resolution

        if not start_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        elif not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        bbox = [
            longitude - bbox_size/2,
            latitude - bbox_size/2,
            longitude + bbox_size/2,
            latitude + bbox_size/2
        ]

        width = height = int(bbox_size * 111000 / resolution)

        payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{start_date}T00:00:00Z",
                            "to": f"{end_date}T23:59:59Z"
                        },
                        "maxCloudCoverage": max_cloud_coverage
                    }
                }]
            },
            "output": {
                "width": width,
                "height": height,
                "responses": [{
                    "identifier": "default",
                    "format": {"type": "image/png"}  # Changed to PNG for better compatibility
                }]
            },
            "evalscript": self.create_evalscript()
        }

        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/v1/process",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                return None

            # Try to open as PNG image
            with io.BytesIO(response.content) as buffer:
                img = Image.open(buffer)
                if img.mode == 'RGBA':
                    return np.array(img)
                return np.array(img.convert('RGBA'))

        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def save_image(self,
                   image_data: np.ndarray,
                   latitude: float,
                   longitude: float,
                   output_dir: str = "sentinel_images",
                   date_str: str = None) -> str:
        """
        Save image data with verification
        """
        os.makedirs(output_dir, exist_ok=True)

        if not date_str:
            date_str = datetime.now().strftime('%Y%m%d')

        filename = f"sentinel_{latitude:.6f}_{longitude:.6f}_{date_str}.npy"
        filepath = os.path.join(output_dir, filename)

        try:
            # Verify image data
            if not isinstance(image_data, np.ndarray) or len(image_data.shape) != 3:
                raise ValueError("Invalid image data format")

            np.save(filepath, {
                'image': image_data,
                'metadata': {
                    'coordinates': (latitude, longitude),
                    'date': date_str,
                    'shape': image_data.shape
                }
            })

            # Save preview
            preview_path = os.path.join(output_dir, f"preview_{latitude:.6f}_{longitude:.6f}_{date_str}.png")
            Image.fromarray(image_data[..., :3]).save(preview_path)

            return filepath
        except Exception as e:
            print(f"Error saving image: {e}")
            return None

    def fetch_multiple_coordinates(self,
                                   coordinates: List[Tuple[float, float]],
                                   output_dir: str = "sentinel_images",
                                   **kwargs) -> List[str]:
        """
        Robust multi-coordinate fetching with progress tracking
        """
        saved_files = []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, (lat, lon) in enumerate(coordinates):
            print(f"\nProcessing {i+1}/{len(coordinates)}: ({lat}, {lon})")

            image_data = self.fetch_image(lat, lon, **kwargs)
            if image_data is None:
                print(f"Failed to fetch image for ({lat}, {lon})")
                continue

            filepath = self.save_image(image_data, lat, lon, output_dir)
            if filepath:
                saved_files.append(filepath)
                print(f"Successfully saved: {filepath}")
            else:
                print(f"Failed to save image for ({lat}, {lon})")

        return saved_files


if __name__ == "__main__":
    # Initialize with your credentials
    fetcher = SentinelHubFetcher(
        client_id="69b0e122-2c43-444c-a03a-11f80f0fa3f6",
        client_secret="UzfLRBn4lWFxz9hypPOeWKxV4BW8LYsT"
    )

    # Test coordinates (start with just one for debugging)
    test_coordinates = [
        (-3.4653, -62.2159),  # Manaus, Brazil
        # (-8.7619, -63.9019),  # Add more after first success
        # (-5.8893, -61.9794),
    ]

    # Fetch with conservative parameters
    results = fetcher.fetch_multiple_coordinates(
        coordinates=test_coordinates,
        bbox_size=0.01,  # Smaller area for testing
        resolution=20,   # More reliable resolution
        max_cloud_coverage=50,  # More lenient cloud coverage
        start_date="2024-01-01",
        end_date="2024-01-10"  # Shorter time range
    )

    print("\nFinal Results:")
    print(f"Successfully processed {len(results)}/{len(test_coordinates)} images")
    for path in results:
        print(f"- {path}")