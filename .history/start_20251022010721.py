#!/usr/bin/env python3
"""
ResuAI Startup Script
Quick setup and launch script for ResuAI application
"""

import os
import sys
import subprocess
import time

def print_header():
    print("=" * 60)
    print("🚀 ResuAI - Personal Data Sheet & Resume Screening System")
    print("   Version 2.0.0 - Excel & CSC Format Support")
    print("=" * 60)
    print()

def check_python():
    """Check Python version"""
    print("✓ Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    print(f"  Python {sys.version.split()[0]} detected")

def check_dependencies():
    """Check if required packages are installed"""
    print("\n✓ Checking dependencies...")
    required_packages = [
        'flask', 'psycopg2', 'pandas', 'openpyxl', 
        'xlrd', 'nltk', 'sklearn', 'bcrypt'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def check_database():
    """Check database connection"""
    print("\n✓ Checking database connection...")
    try:
        from database import DatabaseManager
        db = DatabaseManager()
        print("  ✓ Database connection successful")
        return True
    except Exception as e:
        print(f"  ❌ Database connection failed: {e}")
        return False

def check_admin_user():
    """Check if admin user exists"""
    print("\n✓ Checking admin user...")
    try:
        from database import DatabaseManager
        db = DatabaseManager()
        admin = db.get_user_by_email("admin@resumeai.local")
        if admin:
            print("  ✓ Admin user exists")
            print("  📧 Email: admin@resumeai.local")
            print("  🔑 Password: admin123 (change after login!)")
            return True
        else:
            print("  ⚠️  Admin user not found")
            return False
    except Exception as e:
        print(f"  ❌ Error checking admin user: {e}")
        return False

def create_admin_user():
    """Create admin user if it doesn't exist"""
    print("\n🔧 Creating admin user...")
    try:
        from database import DatabaseManager
        import bcrypt
        
        db = DatabaseManager()
        password = "admin123"
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        user_id = db.create_user(
            email="admin@resumeai.local",
            password=hashed.decode('utf-8'),
            first_name="System",
            last_name="Administrator",
            is_admin=True
        )
        
        print(f"  ✓ Admin user created with ID: {user_id}")
        print("  📧 Email: admin@resumeai.local")
        print("  🔑 Password: admin123")
        print("  ⚠️  Please change the password after first login!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error creating admin user: {e}")
        return False

def setup_environment():
    """Setup environment if .env doesn't exist"""
    if not os.path.exists('.env'):
        print("\n🔧 Creating .env file...")
        env_content = """# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/resumeai

# Security (CHANGE THIS!)
SECRET_KEY=your-very-long-random-secret-key-change-this-in-production

# Development settings
FLASK_ENV=development
FLASK_DEBUG=true
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        print("  ✓ .env file created")
        print("  ⚠️  Please update DATABASE_URL with your PostgreSQL credentials")
        return False
    else:
        print("\n✓ .env file exists")
        return True

def start_application():
    """Start the Flask application"""
    print("\n🚀 Starting ResuAI application...")
    print("   Server will start at: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("\n" + "=" * 60)
    
    # Import and run the app
    try:
        from app import create_app
        app = create_app()
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    print_header()
    
    # Check prerequisites
    check_python()
    
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    if not setup_environment():
        print("\n❌ Please configure your .env file with database credentials")
        sys.exit(1)
    
    if not check_database():
        print("\n❌ Please check your database configuration in .env file")
        print("   Make sure PostgreSQL is running and credentials are correct")
        sys.exit(1)
    
    if not check_admin_user():
        if not create_admin_user():
            print("\n❌ Failed to create admin user")
            sys.exit(1)
    
    print("\n✅ All checks passed! Starting application...")
    time.sleep(2)
    
    start_application()

if __name__ == "__main__":
    main()