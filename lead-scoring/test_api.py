import requests
import json
import time
from typing import Dict, List, Any

def test_lead_scoring_api():
    """Test the Lead Scoring System API"""
    
    base_url = "http://localhost:8004"
    
    # Test data
    test_lead = {
        "lead_id": "TEST001",
        "company_size": "medium",
        "industry": "technology",
        "job_title": "manager",
        "lead_source": "website",
        "engagement_score": 75,
        "website_visits": 5,
        "email_opens": 3,
        "form_submissions": 2,
        "demo_requests": 1,
        "content_downloads": 4,
        "social_media_engagement": 2,
        "days_since_last_activity": 2,
        "budget_range": "medium",
        "authority_level": "influencer",
        "timeline": "3_months",
        "pain_points": ["efficiency", "cost"],
        "competitor_usage": False,
        "marketing_qualified": True,
        "sales_qualified": False
    }
    
    batch_leads = [
        {
            "lead_id": "LEAD001",
            "company_size": "large",
            "industry": "technology",
            "job_title": "executive",
            "lead_source": "referral",
            "engagement_score": 85,
            "website_visits": 10,
            "email_opens": 8,
            "form_submissions": 5,
            "demo_requests": 3,
            "content_downloads": 7,
            "social_media_engagement": 5,
            "days_since_last_activity": 1,
            "budget_range": "large",
            "authority_level": "decision_maker",
            "timeline": "1_month",
            "pain_points": ["efficiency", "scalability", "security"],
            "competitor_usage": False,
            "marketing_qualified": True,
            "sales_qualified": True
        },
        {
            "lead_id": "LEAD002",
            "company_size": "small",
            "industry": "retail",
            "job_title": "analyst",
            "lead_source": "social_media",
            "engagement_score": 45,
            "website_visits": 3,
            "email_opens": 2,
            "form_submissions": 1,
            "demo_requests": 0,
            "content_downloads": 2,
            "social_media_engagement": 3,
            "days_since_last_activity": 15,
            "budget_range": "small",
            "authority_level": "user",
            "timeline": "6_months",
            "pain_points": ["cost"],
            "competitor_usage": True,
            "marketing_qualified": False,
            "sales_qualified": False
        },
        {
            "lead_id": "LEAD003",
            "company_size": "enterprise",
            "industry": "finance",
            "job_title": "director",
            "lead_source": "website",
            "engagement_score": 92,
            "website_visits": 15,
            "email_opens": 12,
            "form_submissions": 8,
            "demo_requests": 4,
            "content_downloads": 10,
            "social_media_engagement": 8,
            "days_since_last_activity": 0,
            "budget_range": "enterprise",
            "authority_level": "influencer",
            "timeline": "immediate",
            "pain_points": ["efficiency", "compliance", "security"],
            "competitor_usage": False,
            "marketing_qualified": True,
            "sales_qualified": True
        }
    ]
    
    print("🧪 Testing Lead Scoring System API")
    print("=" * 50)
    
    try:
        # Test 1: Health Check
        print("\n1. Testing Health Check...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            health_data = response.json()
            print(f"   Status: {health_data['status']}")
            print(f"   Model Loaded: {health_data['model_loaded']}")
            
            # If models not loaded, train them first
            if not health_data['model_loaded']:
                print("\n⚠️ Models not loaded. Training models...")
                train_response = requests.post(f"{base_url}/train")
                if train_response.status_code == 200:
                    print("✅ Models trained successfully")
                else:
                    print("❌ Model training failed")
                    return False
        else:
            print("❌ Health check failed")
            return False
        
        # Test 2: API Info
        print("\n2. Testing API Info...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ API info endpoint working")
            data = response.json()
            print(f"   API: {data['message']}")
            print(f"   Version: {data['version']}")
            print(f"   Models Loaded: {data['models_loaded']}")
        else:
            print("❌ API info endpoint failed")
            return False
        
        # Test 3: Single Lead Scoring
        print("\n3. Testing Single Lead Scoring...")
        response = requests.post(f"{base_url}/score", json=test_lead)
        if response.status_code == 200:
            print("✅ Single lead scoring successful")
            result = response.json()
            print(f"   Lead ID: {result['lead_id']}")
            print(f"   Score: {result['score']}/100")
            print(f"   Conversion Probability: {result['conversion_probability']:.1%}")
            print(f"   Model Used: {result['model_used']}")
            print(f"   Recommendation: {result['recommendation']}")
        else:
            print("❌ Single lead scoring failed")
            print(f"   Error: {response.json().get('detail', 'Unknown error')}")
            return False
        
        # Test 4: Batch Lead Scoring
        print("\n4. Testing Batch Lead Scoring...")
        response = requests.post(f"{base_url}/score/batch", json=batch_leads)
        if response.status_code == 200:
            print("✅ Batch lead scoring successful")
            results = response.json()
            print(f"   Total Processed: {results['total_processed']}")
            print(f"   High Priority: {results['high_priority']}")
            print(f"   Medium Priority: {results['medium_priority']}")
            print(f"   Low Priority: {results['low_priority']}")
            print(f"   Processing Time: {results['processing_time_ms']:.1f}ms")
            
            # Show individual scores
            for score in results['scores']:
                print(f"   {score['lead_id']}: Score={score['score']}, Prob={score['conversion_probability']:.1%}")
        else:
            print("❌ Batch lead scoring failed")
            print(f"   Error: {response.json().get('detail', 'Unknown error')}")
            return False
        
        # Test 5: Model Info
        print("\n5. Testing Model Info...")
        response = requests.get(f"{base_url}/model/info")
        if response.status_code == 200:
            print("✅ Model info endpoint working")
            model_info = response.json()
            print(f"   Model Name: {model_info['model_name']}")
            print(f"   Version: {model_info['version']}")
            print(f"   Algorithms: {', '.join(model_info['algorithms'])}")
            print(f"   Features: {len(model_info['features'])}")
            print(f"   Training Samples: {model_info['training_samples']}")
        else:
            print("❌ Model info failed")
            return False
        
        # Test 6: Model Performance
        print("\n6. Testing Model Performance...")
        response = requests.get(f"{base_url}/model/performance")
        if response.status_code == 200:
            print("✅ Model performance endpoint working")
            performance = response.json()
            
            for model_name, metrics in performance.items():
                print(f"   {model_name.upper()}:")
                print(f"     Accuracy: {metrics['accuracy']:.3f}")
                print(f"     Precision: {metrics['precision']:.3f}")
                print(f"     Recall: {metrics['recall']:.3f}")
                print(f"     F1-Score: {metrics['f1_score']:.3f}")
                print(f"     AUC-ROC: {metrics['auc_roc']:.3f}")
        else:
            print("❌ Model performance failed")
            return False
        
        # Test 7: Generate Sample Data
        print("\n7. Testing Sample Data Generation...")
        response = requests.post(f"{base_url}/generate-sample-data", json={"n_leads": 5})
        if response.status_code == 200:
            print("✅ Sample data generation successful")
            sample_data = response.json()
            print(f"   Generated {sample_data['total_generated']} sample leads")
            print(f"   Avg Engagement Score: {sample_data['summary_stats']['avg_engagement_score']:.1f}")
            print(f"   Avg Website Visits: {sample_data['summary_stats']['avg_website_visits']:.1f}")
            print(f"   Avg Email Opens: {sample_data['summary_stats']['avg_email_opens']:.1f}")
        else:
            print("❌ Sample data generation failed")
            return False
        
        # Test 8: Test Different Models
        print("\n8. Testing Different Models...")
        models_to_test = ["random_forest", "logistic_regression"]
        
        try:
            import xgboost
            models_to_test.append("xgboost")
        except ImportError:
            print("⚠️ XGBoost not available, skipping XGBoost test")
        
        for model_type in models_to_test:
            response = requests.post(f"{base_url}/score?model_type={model_type}", json=test_lead)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {model_type}: Score={result['score']}, Prob={result['conversion_probability']:.1%}")
            else:
                print(f"❌ {model_type} failed: {response.json().get('detail', 'Unknown error')}")
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Lead Scoring System is working correctly.")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API server")
        print("   Make sure the server is running on port 8004")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

def test_model_training():
    """Test model training functionality"""
    
    base_url = "http://localhost:8004"
    
    print("\n🧪 Testing Model Training")
    print("=" * 30)
    
    try:
        # Test training with custom data
        custom_training_data = [
            {
                "lead_id": "TRAIN001",
                "company_size": "large",
                "industry": "technology",
                "job_title": "executive",
                "lead_source": "website",
                "engagement_score": 90,
                "website_visits": 20,
                "email_opens": 15,
                "form_submissions": 8,
                "demo_requests": 5,
                "content_downloads": 12,
                "social_media_engagement": 10,
                "days_since_last_activity": 1,
                "budget_range": "large",
                "authority_level": "decision_maker",
                "timeline": "1_month",
                "pain_points": ["efficiency", "scalability"],
                "competitor_usage": False,
                "marketing_qualified": True,
                "sales_qualified": True
            }
        ]
        
        # Generate more training data
        for i in range(2, 21):
            lead_data = custom_training_data[0].copy()
            lead_data["lead_id"] = f"TRAIN{i:03d}"
            lead_data["engagement_score"] = max(0, 90 - i * 4)
            lead_data["website_visits"] = max(0, 20 - i)
            lead_data["email_opens"] = max(0, 15 - i)
            custom_training_data.append(lead_data)
        
        response = requests.post(f"{base_url}/train", json=custom_training_data)
        if response.status_code == 200:
            print("✅ Model training with custom data successful")
            result = response.json()
            print(f"   Best Model: {result['model_type']}")
            print(f"   Accuracy: {result['accuracy']:.3f}")
            print(f"   Precision: {result['precision']:.3f}")
            print(f"   Recall: {result['recall']:.3f}")
            print(f"   F1-Score: {result['f1_score']:.3f}")
            print(f"   AUC-ROC: {result['auc_roc']:.3f}")
            print(f"   Training Samples: {result['training_samples']}")
            return True
        else:
            print("❌ Model training failed")
            print(f"   Error: {response.json().get('detail', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Model training test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run all tests
    success = test_lead_scoring_api()
    
    if success:
        # Run additional training test
        test_model_training()
        
        print("\n🎯 Testing completed!")
        print("\nTo run the API server:")
        print("uvicorn app.main:app --host 0.0.0.0 --port 8004")
    else:
        print("\n❌ Some tests failed. Please check the API server and try again.")
        exit(1)