#!/usr/bin/env python3
"""
Test script to verify the project setup and dependencies.
"""

import os
import sys
import importlib
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required packages can be imported."""
    logger.info("Testing package imports...")
    
    required_packages = [
        'mlx',
        'mlx_lm',
        'pandas',
        'openpyxl',
        'fitz',  # PyMuPDF
        'requests',
        'PIL',
        'numpy',
        'tqdm',
        'jsonlines',
        'regex'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package}")
        except ImportError as e:
            logger.error(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        logger.error(f"Failed to import: {failed_imports}")
        return False
    else:
        logger.info("All packages imported successfully!")
        return True

def test_project_structure():
    """Test that the project structure is correct."""
    logger.info("Testing project structure...")
    
    required_files = [
        'finetune.py',
        'parse_data.py',
        'prepare_dataset.py',
        'requirements.txt',
        'README.md'
    ]
    
    required_dirs = [
        'output'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            logger.info(f"✓ {file}")
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
        else:
            logger.info(f"✓ {dir_name}/")
    
    if missing_files or missing_dirs:
        logger.error(f"Missing files: {missing_files}")
        logger.error(f"Missing directories: {missing_dirs}")
        return False
    else:
        logger.info("Project structure is correct!")
        return True

def test_data_directory():
    """Test that the data directory exists."""
    logger.info("Testing data directory...")
    
    data_dir = os.path.expanduser("~/Desktop/trainingdata/")
    
    if os.path.exists(data_dir):
        logger.info(f"✓ Data directory exists: {data_dir}")
        
        # Check for supported file types
        supported_extensions = {'.pdf', '.xlsx', '.txt', '.csv'}
        found_files = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_extensions:
                    found_files.append(os.path.join(root, file))
        
        if found_files:
            logger.info(f"✓ Found {len(found_files)} supported files")
            for file in found_files[:5]:  # Show first 5 files
                logger.info(f"  - {os.path.basename(file)}")
            if len(found_files) > 5:
                logger.info(f"  ... and {len(found_files) - 5} more")
        else:
            logger.warning("⚠ No supported files found in data directory")
            
        return True
    else:
        logger.error(f"✗ Data directory not found: {data_dir}")
        logger.info("Please create the directory and add your training data")
        return False

def test_model_paths():
    """Test common model paths."""
    logger.info("Testing model paths...")
    
    common_paths = [
        "~/models/mlabonne/gemma-3-27B-it-abliterated-GGUF/gemma-3-27B-it-abliterated.q4_k_m.gguf",
        "~/models/mlx-community/gemma-3-27B-it-qat-4bit",
        "~/models/gemma-3-vision-latex.Q4_K_S.gguf"
    ]
    
    found_models = []
    
    for path in common_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            logger.info(f"✓ Found model: {path}")
            found_models.append(path)
        else:
            logger.warning(f"⚠ Model not found: {path}")
    
    if found_models:
        logger.info(f"Found {len(found_models)} model(s)")
        return True
    else:
        logger.error("No models found in common paths")
        logger.info("Please ensure you have the required models downloaded")
        return False

def test_lm_studio_api():
    """Test LM Studio API connection."""
    logger.info("Testing LM Studio API...")
    
    try:
        import requests
        
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        
        if response.status_code == 200:
            logger.info("✓ LM Studio API is running")
            return True
        else:
            logger.warning(f"⚠ LM Studio API returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.warning("⚠ LM Studio API not accessible (localhost:1234)")
        logger.info("Please start LM Studio and load a vision model")
        return False
    except Exception as e:
        logger.error(f"✗ Error testing LM Studio API: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Running project setup tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Data Directory", test_data_directory),
        ("Model Paths", test_model_paths),
        ("LM Studio API", test_lm_studio_api)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Your setup is ready for fine-tuning.")
        logger.info("\nNext steps:")
        logger.info("1. Ensure your training data is in ~/Desktop/trainingdata/")
        logger.info("2. Start LM Studio with a vision model loaded")
        logger.info("3. Run: python finetune.py --model_path /path/to/your/model")
    else:
        logger.error("❌ Some tests failed. Please fix the issues above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main() 