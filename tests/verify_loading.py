import sys
import os
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Also add scripts to path to import run_pipeline
sys.path.append(os.path.join(project_root, "scripts"))

# We need to mock get_logger because it might try to write to a log file in a restricted location or just fail
from unittest.mock import MagicMock
import llm_data_wash.utils.logger
llm_data_wash.utils.logger.get_logger = MagicMock()

# Now import the function
from run_pipeline import load_local_dataset

def test_loading():
    dummy_path = os.path.join(os.path.dirname(__file__), "dummy_vqav2.json")
    dummy_data = {
        "info": "dummy",
        "questions": [
            {"image_id": 524291, "question": "What is in the person's hand?", "question_id": 524291000},
            {"image_id": 12345, "question": "Is this a test?", "question_id": 12345000}
        ]
    }
    
    os.makedirs("tests", exist_ok=True)
    with open(dummy_path, "w") as f:
        json.dump(dummy_data, f)
        
    config = {
        "data": {
            "image_dir": "/tmp/images",
            "image_prefix": "COCO_train2014_"
        }
    }
    
    try:
        dataset = load_local_dataset(dummy_path, config)
        print(f"Loaded {len(dataset)} items")
        print("First item:", dataset[0])
        
        expected_path = "/tmp/images/COCO_train2014_000000524291.jpg"
        actual_path = dataset[0]["conversations"][0]["image_path"]
        assert actual_path == expected_path, f"Expected {expected_path}, got {actual_path}"
        
        # Check second item
        expected_path_2 = "/tmp/images/COCO_train2014_000000012345.jpg"
        actual_path_2 = dataset[1]["conversations"][0]["image_path"]
        assert actual_path_2 == expected_path_2, f"Expected {expected_path_2}, got {actual_path_2}"
        
        print("Verification passed!")
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
            
if __name__ == "__main__":
    test_loading()
