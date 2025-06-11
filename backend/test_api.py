#!/usr/bin/env python3
"""
Test script for the stock screener API
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000/api"

def test_health():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_presets():
    """Test the presets endpoint"""
    print("\nTesting presets endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/screen/presets")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Found {len(data.get('presets', {}))} presets")
        for preset_name, preset_data in data.get('presets', {}).items():
            print(f"- {preset_name}: {preset_data['name']}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_screening():
    """Test the stock screening endpoint"""
    print("\nTesting stock screening...")
    try:
        # Test with value stocks criteria
        criteria = {
            "criteria": {
                "max_pe_ratio": 20,
                "min_market_cap": 1000000000,
                "min_volume": 500000
            },
            "limit": 10
        }
        
        print("Sending screening request...")
        response = requests.post(f"{BASE_URL}/screen", 
                               json=criteria,
                               headers={'Content-Type': 'application/json'})
        
        print(f"Status: {response.status_code}")
        data = response.json()
        
        if data.get('success'):
            print(f"Found {data.get('count', 0)} matching stocks")
            for stock in data.get('results', [])[:3]:  # Show first 3 results
                print(f"- {stock.get('symbol')}: {stock.get('name')} "
                      f"(PE: {stock.get('pe_ratio')}, Market Cap: {stock.get('market_cap')})")
        else:
            print(f"Error: {data.get('error')}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_stock_details():
    """Test the stock details endpoint"""
    print("\nTesting stock details...")
    try:
        symbol = "AAPL"
        response = requests.get(f"{BASE_URL}/stock/{symbol}")
        print(f"Status: {response.status_code}")
        data = response.json()
        
        if data.get('success'):
            stock = data.get('stock', {})
            print(f"Stock: {stock.get('symbol')} - {stock.get('name')}")
            print(f"Price: ${stock.get('current_price')}")
            print(f"Market Cap: {stock.get('market_cap')}")
            print(f"P/E Ratio: {stock.get('pe_ratio')}")
        else:
            print(f"Error: {data.get('error')}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("Stock Screener API Test Suite")
    print("=" * 40)
    
    tests = [
        ("Health Check", test_health),
        ("Presets", test_presets),
        ("Stock Screening", test_screening),
        ("Stock Details", test_stock_details)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        success = test_func()
        results.append((test_name, success))
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 40)
    print("Test Results:")
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")

if __name__ == "__main__":
    main()
