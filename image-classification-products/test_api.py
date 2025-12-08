import requests
import json
import base64
import numpy as np
from PIL import Image
import io
import time

def create_test_image(category="electronics", size=(224, 224)):
    """Create a test image for classification"""
    if category == "electronics":
        # Simulate electronics image - darker, more structured
        image_array = np.random.randint(50, 150, (*size, 3), dtype=np.uint8)
        # Add some geometric patterns
        for _ in range(5):
            x, y = np.random.randint(0, size[0]-50, 2)
            w, h = np.random.randint(20, 50, 2)
            image_array[y:y+h, x:x+w] = np.random.randint(100, 255, (h, w, 3))
    
    elif category == "clothing":
        # Simulate clothing image - softer colors, fabric texture
        base_color = np.random.randint(100, 200, 3)
        image_array = np.full((*size, 3), base_color, dtype=np.uint8)
        # Add fabric texture
        noise = np.random.randint(-30, 30, (*size, 3))
        image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    
    elif category == "food":
        # Simulate food image - warmer colors
        base_color = np.random.randint([150, 100, 50], [255, 200, 150])
        image_array = np.full((*size, 3), base_color, dtype=np.uint8)
        # Add some variation using NumPy instead of cv2
        for _ in range(3):
            x, y = np.random.randint(0, size[0]-30, 2)
            radius = np.random.randint(10, 30)
            # Create circular pattern using NumPy
            y_grid, x_grid = np.ogrid[:size[0], :size[1]]
            mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
            image_array[mask] = np.random.randint(100, 255, 3)
    
    else:
        # Default random image
        image_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    
    # Convert to PIL Image
    image = Image.fromarray(image_array)
    return image

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get("http://localhost:8008/health")
        print(f"Health check status: {response.status_code}")
        print(f"Health check response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_classify_image():
    """Test single image classification"""
    print("\nTesting single image classification...")
    
    try:
        # Create test image
        test_image = create_test_image("electronics")
        image_base64 = image_to_base64(test_image)
        
        # Prepare request
        request_data = {
            "image_data": image_base64,
            "model_type": "efficientnet_b0",
            "top_k": 5,
            "confidence_threshold": 0.1,
            "image_size": "small"
        }
        
        # Send request
        response = requests.post(
            "http://localhost:8008/classify",
            json=request_data
        )
        
        print(f"Classification status: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"Predictions: {result.get('predictions', [])}")
            print(f"Processing time: {result.get('processing_time', 0):.3f}s")
            print(f"Model used: {result.get('model_type', 'unknown')}")
            return True
        else:
            print(f"Classification error: {result}")
            return False
            
    except Exception as e:
        print(f"Classification test failed: {e}")
        return False

def test_batch_classification():
    """Test batch image classification"""
    print("\nTesting batch image classification...")
    
    try:
        # Create multiple test images
        images = []
        categories = ["electronics", "clothing", "food", "electronics", "clothing"]
        
        for category in categories:
            test_image = create_test_image(category)
            image_base64 = image_to_base64(test_image)
            images.append(image_base64)
        
        # Prepare request
        request_data = {
            "images": images,
            "model_type": "efficientnet_b0",
            "top_k": 3,
            "confidence_threshold": 0.1,
            "image_size": "small"
        }
        
        # Send request
        response = requests.post(
            "http://localhost:8008/classify/batch",
            json=request_data
        )
        
        print(f"Batch classification status: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"Total images: {result.get('total_images', 0)}")
            print(f"Successful: {result.get('successful_classifications', 0)}")
            print(f"Total processing time: {result.get('total_processing_time', 0):.3f}s")
            print(f"Average processing time: {result.get('average_processing_time', 0):.3f}s")
            
            # Show first result
            results = result.get('results', [])
            if results:
                first_result = results[0]
                print(f"First image predictions: {first_result.get('predictions', [])}")
            
            return True
        else:
            print(f"Batch classification error: {result}")
            return False
            
    except Exception as e:
        print(f"Batch classification test failed: {e}")
        return False

def test_get_models():
    """Test get available models"""
    print("\nTesting get available models...")
    
    try:
        response = requests.get("http://localhost:8008/models")
        print(f"Get models status: {response.status_code}")
        
        if response.status_code == 200:
            models = response.json()
            print(f"Available models: {len(models)}")
            for model in models[:2]:  # Show first 2 models
                print(f"- {model.get('model_type', 'unknown')}: {model.get('description', 'no description')}")
            return True
        else:
            print(f"Get models error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Get models test failed: {e}")
        return False

def test_generate_sample_data():
    """Test sample data generation"""
    print("\nTesting sample data generation...")
    
    try:
        # This would require a dedicated endpoint or we can create sample data locally
        # For now, we'll create some sample images manually
        sample_images = []
        categories = ["electronics", "clothing", "food", "books", "home_kitchen"]
        
        for category in categories[:3]:  # Create 3 sample images
            test_image = create_test_image(category)
            image_base64 = image_to_base64(test_image)
            sample_images.append({
                "image_data": image_base64,
                "category": category,
                "label": category.replace('_', ' ').title()
            })
        
        print(f"Generated {len(sample_images)} sample images")
        print("Sample data generation test completed successfully")
        return True
        
    except Exception as e:
        print(f"Sample data generation test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("Starting Image Classification API Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Single Image Classification", test_classify_image),
        ("Batch Classification", test_batch_classification),
        ("Get Models", test_get_models),
        ("Sample Data Generation", test_generate_sample_data)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"{test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"{test_name}: FAILED - {e}")
            results.append((test_name, False))
        
        # Small delay between tests
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        print(f"{test_name}: {'✓' if success else '✗'}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - passed} tests failed")
    
    return passed == total

if __name__ == "__main__":
    # Wait a moment for the server to be ready
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    success = run_all_tests()
    exit(0 if success else 1)