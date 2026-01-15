import requests
import json
import base64
import numpy as np
import time
from PIL import Image
import io
import random

def create_test_image(product_type="electronics", size=(224, 224)):
    """Create a test image for quality control"""
    # Create base image with product-appropriate colors
    if product_type == "electronics":
        # Electronics - dark colors, metallic appearance
        base_color = np.random.randint([50, 50, 50], [100, 100, 100])
        image_array = np.full((*size, 3), base_color, dtype=np.uint8)
        
        # Add some metallic shine
        for _ in range(2):
            x, y = np.random.randint(0, size[0]-50, 2)
            width, height = np.random.randint(20, 50, 2)
            image_array[y:y+height, x:x+width] = np.random.randint([150, 150, 150], [200, 200, 200])
    
    elif product_type == "automotive":
        # Automotive - painted surfaces
        base_color = np.random.randint([100, 100, 150], [200, 200, 250])
        image_array = np.full((*size, 3), base_color, dtype=np.uint8)
        
        # Add some reflections
        for _ in range(3):
            x, y = np.random.randint(0, size[0]-30, 2)
            radius = np.random.randint(10, 30)
            y_grid, x_grid = np.ogrid[:size[0], :size[1]]
            mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
            image_array[mask] = np.clip(image_array[mask] + 50, 0, 255)
    
    elif product_type == "textiles":
        # Textiles - fabric texture
        base_color = np.random.randint([100, 80, 60], [200, 180, 160])
        image_array = np.full((*size, 3), base_color, dtype=np.uint8)
        
        # Add fabric texture pattern
        for i in range(0, size[0], 10):
            image_array[i:i+2, :] = np.clip(image_array[i:i+2, :] + np.random.randint(-20, 20), 0, 255)
    
    elif product_type == "food":
        # Food - natural colors
        base_color = np.random.randint([150, 100, 50], [255, 200, 150])
        image_array = np.full((*size, 3), base_color, dtype=np.uint8)
        
        # Add some variation
        for _ in range(5):
            x, y = np.random.randint(0, size[0]-20, 2)
            radius = np.random.randint(5, 20)
            y_grid, x_grid = np.ogrid[:size[0], :size[1]]
            mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
            color_variation = np.random.randint(-30, 30, 3)
            image_array[mask] = np.clip(image_array[mask] + color_variation, 0, 255)
    
    else:
        # Default - generic product
        base_color = np.random.randint([100, 100, 100], [200, 200, 200])
        image_array = np.full((*size, 3), base_color, dtype=np.uint8)
    
    # Add some defects randomly for testing
    if random.random() < 0.3:  # 30% chance of defects
        defect_type = random.choice(["scratch", "discoloration", "dent"])
        
        if defect_type == "scratch":
            # Add scratch-like defect
            x1, y1 = np.random.randint(0, size[0]-50, 2)
            x2, y2 = x1 + np.random.randint(20, 50), y1 + np.random.randint(-10, 10)
            if 0 <= x2 < size[0] and 0 <= y2 < size[1]:
                cv2.line(image_array, (x1, y1), (x2, y2), (0, 0, 0), 2)
        
        elif defect_type == "discoloration":
            # Add discoloration
            x, y = np.random.randint(0, size[0]-30, 2)
            radius = np.random.randint(10, 30)
            y_grid, x_grid = np.ogrid[:size[0], :size[1]]
            mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
            discoloration = np.random.randint([-50, -50, -50], [50, 50, 50], 3)
            image_array[mask] = np.clip(image_array[mask] + discoloration, 0, 255)
        
        elif defect_type == "dent":
            # Add dent (darker area)
            x, y = np.random.randint(0, size[0]-40, 2)
            radius = np.random.randint(15, 40)
            y_grid, x_grid = np.ogrid[:size[0], :size[1]]
            mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
            image_array[mask] = np.clip(image_array[mask] - 30, 0, 255)
    
    # Convert to PIL Image
    image = Image.fromarray(image_array)
    
    return image

def image_to_base64(image, format="JPEG"):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def test_health_check():
    """Test health check endpoint"""
    print("🩺 Testing health check...")
    
    try:
        response = requests.get("http://localhost:8003/health")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Total inspected: {data['total_inspected']}")
            print(f"   Defects found: {data['defects_found']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_single_product_inspection():
    """Test single product inspection"""
    print("\n🔍 Testing single product inspection...")
    
    try:
        # Create test image
        test_image = create_test_image("electronics")
        image_base64 = image_to_base64(test_image)
        
        # Prepare request
        request_data = {
            "product_id": "laptop_001",
            "product_name": "Gaming Laptop Pro",
            "category": "electronics",
            "image_data": f"data:image/jpeg;base64,{image_base64}",
            "image_format": "jpeg",
            "batch_id": "batch_001",
            "manufacturing_date": "2024-01-15T10:00:00"
        }
        
        response = requests.post(
            "http://localhost:8003/inspect-product",
            json=request_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Single product inspection passed")
            print(f"   Product ID: {data['product_id']}")
            print(f"   Status: {data['status']}")
            print(f"   Quality Score: {data['quality_score']:.3f}")
            print(f"   Defects Detected: {len(data['defects_detected'])}")
            print(f"   Processing Time: {data['processing_time']:.3f}s")
            
            if data['defects_detected']:
                print("   Defect Details:")
                for defect in data['defects_detected']:
                    print(f"     - {defect['defect_type']}: {defect['confidence']:.3f} confidence")
            
            return True
        else:
            print(f"❌ Single product inspection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Single product inspection error: {e}")
        return False

def test_batch_inspection():
    """Test batch inspection"""
    print("\n📦 Testing batch inspection...")
    
    try:
        products = []
        categories = ["electronics", "automotive", "textiles", "food"]
        
        for i in range(5):
            # Create test image
            category = categories[i % len(categories)]
            test_image = create_test_image(category)
            image_base64 = image_to_base64(test_image)
            
            product = {
                "product_id": f"product_{i+1:03d}",
                "product_name": f"Test Product {i+1}",
                "category": category,
                "image_data": f"data:image/jpeg;base64,{image_base64}",
                "image_format": "jpeg",
                "batch_id": "batch_002"
            }
            products.append(product)
        
        request_data = {
            "products": products,
            "batch_config": {
                "parallel_processing": True,
                "quality_threshold": 0.8
            }
        }
        
        response = requests.post(
            "http://localhost:8003/inspect-batch",
            json=request_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch inspection passed")
            print(f"   Batch ID: {data['batch_id']}")
            print(f"   Total Products: {data['total_products']}")
            print(f"   Passed: {data['passed_products']}")
            print(f"   Failed: {data['failed_products']}")
            print(f"   Warnings: {data['warning_products']}")
            print(f"   Processing Time: {data['processing_time']:.3f}s")
            print(f"   Defect Rate: {data['batch_summary']['defect_rate']:.3f}")
            print(f"   Average Quality Score: {data['batch_summary']['average_quality_score']:.3f}")
            
            return True
        else:
            print(f"❌ Batch inspection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch inspection error: {e}")
        return False

def test_model_info():
    """Test model information endpoint"""
    print("\nℹ️ Testing model information...")
    
    try:
        response = requests.get("http://localhost:8003/model-info")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Model Name: {data['model_name']}")
            print(f"   Version: {data['version']}")
            print(f"   Accuracy: {data['accuracy']:.3f}")
            print(f"   Supported Defects: {len(data['defect_types'])}")
            print(f"   Supported Categories: {len(data['supported_categories'])}")
            
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_quality_standards():
    """Test quality standards endpoint"""
    print("\n📋 Testing quality standards...")
    
    try:
        # Test all standards
        response = requests.get("http://localhost:8003/quality-standards")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Quality standards retrieved")
            print(f"   Categories: {list(data.keys())}")
            
            # Test specific category
            response2 = requests.get("http://localhost:8003/quality-standards?category=electronics")
            if response2.status_code == 200:
                electronics_data = response2.json()
                print(f"   Electronics Standards:")
                print(f"     Max Defects: {electronics_data['max_defects']}")
                print(f"     Min Quality Score: {electronics_data['min_quality_score']}")
                print(f"     Critical Defect Threshold: {electronics_data['critical_defect_threshold']}")
            
            return True
        else:
            print(f"❌ Quality standards failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Quality standards error: {e}")
        return False

def test_statistics():
    """Test statistics endpoint"""
    print("\n📊 Testing statistics...")
    
    try:
        response = requests.get("http://localhost:8003/statistics")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Statistics retrieved")
            print(f"   Total Inspected: {data['total_inspected']}")
            print(f"   Defects Found: {data['defects_found']}")
            print(f"   Defect Rate: {data['defect_rate']:.3f}")
            print(f"   Model Version: {data['model_version']}")
            
            return True
        else:
            print(f"❌ Statistics failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Statistics error: {e}")
        return False

def test_upload_image():
    """Test image upload endpoint"""
    print("\n📤 Testing image upload...")
    
    try:
        # Create test image
        test_image = create_test_image("automotive")
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Prepare files for upload
        files = {'file': ('test_image.jpg', img_byte_arr, 'image/jpeg')}
        
        # Prepare form data
        data = {
            'product_id': 'car_part_001',
            'product_name': 'Car Door Panel',
            'category': 'automotive',
            'batch_id': 'batch_003'
        }
        
        response = requests.post(
            "http://localhost:8003/upload-image",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result_data = response.json()
            print(f"✅ Image upload passed")
            print(f"   Product ID: {result_data['product_id']}")
            print(f"   Status: {result_data['status']}")
            print(f"   Quality Score: {result_data['quality_score']:.3f}")
            print(f"   Defects Detected: {len(result_data['defects_detected'])}")
            
            return True
        else:
            print(f"❌ Image upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Image upload error: {e}")
        return False

def test_generate_sample_data():
    """Test sample data generation"""
    print("\n🎲 Testing sample data generation...")
    
    try:
        response = requests.post("http://localhost:8003/generate-sample-data?n_samples=5")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Sample data generated")
            print(f"   Count: {data['count']}")
            print(f"   Generated at: {data['generated_at']}")
            
            return True
        else:
            print(f"❌ Sample data generation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Sample data generation error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🧪 Running Quality Control CV API Tests")
    print("=" * 60)
    
    tests = [
        test_health_check,
        test_model_info,
        test_quality_standards,
        test_statistics,
        test_single_product_inspection,
        test_batch_inspection,
        test_upload_image,
        test_generate_sample_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(0.5)  # Small delay between tests
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Quality Control CV API is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the API status.")
        return False

if __name__ == "__main__":
    # Note: Make sure the API server is running before executing tests
    print("⚠️  Note: Ensure the Quality Control CV API server is running on port 8003")
    print("   Start with: python -m app.main")
    print()
    
    success = run_all_tests()
    exit(0 if success else 1)