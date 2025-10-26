#!/usr/bin/env python3
"""
Test the new notification and help systems
"""

import requests
import json
import time

def test_notification_system():
    """Test notification system functionality"""
    print("🧪 Testing Notification & Help Systems")
    print("=" * 50)
    
    # Test system status endpoint
    try:
        response = requests.get("http://127.0.0.1:5000/api/system/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ System Status API: Working")
            print(f"   Database: {'✅' if data['data']['database'] else '❌'}")
            print(f"   Analytics: {'✅' if data['data']['analytics'] else '❌'}")
            print(f"   Upload: {'✅' if data['data']['upload'] else '❌'}")
            print(f"   Assessment: {'✅' if data['data']['assessment'] else '❌'}")
        else:
            print(f"❌ System Status API: Error {response.status_code}")
    except Exception as e:
        print(f"❌ System Status API: {e}")
    
    # Test dashboard accessibility
    try:
        response = requests.get("http://127.0.0.1:5000/dashboard")
        if response.status_code == 200:
            print("✅ Dashboard: Accessible")
            
            # Check if our new CSS and JS files are referenced
            content = response.text
            if 'notifications.css' in content:
                print("✅ Notification CSS: Included")
            else:
                print("❌ Notification CSS: Missing")
                
            if 'help-system.css' in content:
                print("✅ Help System CSS: Included")
            else:
                print("❌ Help System CSS: Missing")
                
            if 'notifications.js' in content:
                print("✅ Notification JS: Included")
            else:
                print("❌ Notification JS: Missing")
                
            if 'help-system.js' in content:
                print("✅ Help System JS: Included")
            else:
                print("❌ Help System JS: Missing")
                
            # Check for help attributes
            if 'data-help=' in content:
                print("✅ Help Attributes: Found")
            else:
                print("❌ Help Attributes: Missing")
                
        else:
            print(f"❌ Dashboard: Error {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard: {e}")
    
    print("\n🎯 Test Summary:")
    print("- Notification system CSS and JS should be loaded")
    print("- Help system CSS and JS should be loaded")
    print("- Help attributes should be present on key elements")
    print("- System status monitoring should be working")
    print("\n🚀 Ready for user testing!")

if __name__ == "__main__":
    test_notification_system()