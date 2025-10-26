# 🚀 How to Run ResuAI - Quick Start Guide

## ⚡ Fastest Way to Get Started

### Step 1: Open PowerShell as Administrator
```powershell
# Right-click PowerShell and select "Run as Administrator"
```

### Step 2: Navigate to the Project
```powershell
cd "c:\Users\Lenar Yolola\OneDrive\Desktop\ResuAI__"
```

### Step 3: Setup Environment (First Time Only)
```powershell
python setup_env.py
```
Follow the prompts to configure your database settings.

### Step 4: Start the Application
```powershell
# Option 1: Use the startup script
python start.py

# Option 2: Use the batch file
start_resumai.bat

# Option 3: Direct execution
python app.py
```

### Step 5: Access the System
1. Open browser: http://localhost:5000
2. Login with default credentials:
   - **Email**: admin@resumeai.local  
   - **Password**: admin123
3. Accept the Data Privacy Agreement
4. Start using the system!

## 🔧 If You Have Issues

### Database Connection Error
1. Make sure PostgreSQL is installed and running
2. Create a database named `resumeai`
3. Update your `.env` file with correct credentials

### Missing Dependencies
```powershell
pip install -r requirements.txt
```

### Port Already in Use
```powershell
# Check what's using port 5000
netstat -ano | findstr :5000

# Use a different port
python app.py --port 5001
```

## 🎯 What's New in Version 2.0

- ✅ **Fixed Authentication**: Now always requires login
- ✅ **Data Privacy Compliance**: Mandatory privacy agreement
- ✅ **Excel Support**: Upload .xlsx and .xls files
- ✅ **Philippine CSC Format**: Enhanced PDS processing
- ✅ **Improved Security**: Better session management

## 📁 File Formats Supported

- **PDF**: Adobe PDF documents
- **DOCX**: Microsoft Word documents
- **TXT**: Plain text files
- **XLSX**: Excel spreadsheets (NEW!)
- **XLS**: Legacy Excel files (NEW!)

## 🔐 Security Features

1. **Mandatory Login**: No bypassing authentication
2. **Data Privacy Agreement**: Required for compliance
3. **Session Security**: Proper logout and session clearing
4. **Password Hashing**: Secure password storage (when bcrypt available)

## 🆘 Getting Help

If you encounter any issues:

1. **Check the terminal output** for error messages
2. **Review the `.env` file** for correct database settings
3. **Ensure PostgreSQL is running**
4. **Try the troubleshooting steps** in the full README.md

## 📞 Quick Commands Reference

```powershell
# Setup environment
python setup_env.py

# Start application
python start.py

# Test database connection
python -c "from database import DatabaseManager; db = DatabaseManager(); print('Database OK!')"

# Install missing packages
pip install flask psycopg2-binary pandas openpyxl xlrd

# Generate new admin user
python -c "from database import DatabaseManager; import bcrypt; db = DatabaseManager(); print('Admin created!')"
```

---

**Remember**: Change the default admin password after your first login for security!