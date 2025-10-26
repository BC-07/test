#!/usr/bin/env python3
"""
Test the exact utils.py process_excel_pds_file method
"""

def test_exact_utils_process():
    """Test the exact method that's called from the app"""
    print("🔍 TESTING EXACT UTILS PROCESS")
    print("=" * 50)
    
    try:
        from utils import PersonalDataSheetProcessor
        processor = PersonalDataSheetProcessor()
        
        # Simulate the exact call from app.py
        test_file = "Sample PDS New.xlsx"
        filename = "Sample PDS New.xlsx"
        job = {'id': 1, 'title': 'Test Job'}  # Mock job object
        
        print(f"📄 Testing process_excel_pds_file: {filename}")
        result = processor.process_excel_pds_file(test_file, filename, job)
        
        if result:
            print("✅ process_excel_pds_file returned data!")
            print(f"👤 Name: {result.get('name', 'MISSING')}")
            print(f"📧 Email: {result.get('email', 'MISSING')}")
            print(f"📱 Phone: {result.get('phone', 'MISSING')}")
            print(f"🏷️ Processing Type: {result.get('processing_type', 'MISSING')}")
            print(f"📊 Score: {result.get('score', 'MISSING')}")
            
            # Check PDS data
            if 'pds_extracted_data' in result:
                pds_data = result['pds_extracted_data']
                print(f"📋 PDS Data Sections: {list(pds_data.keys())}")
                
            print(f"\n🔍 All fields in result:")
            for key, value in result.items():
                if isinstance(value, (list, dict)):
                    print(f"   {key}: [{type(value).__name__} with {len(value)} items]")
                else:
                    print(f"   {key}: {value}")
                    
            return True
        else:
            print("❌ process_excel_pds_file returned None!")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_exact_utils_process()