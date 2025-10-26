# ResuAI - AI-Powered Resume Screening System

ResuAI is an advanced resume screening and job recommendation web application powered by machine learning and BERT-based semantic analysis. It helps automate the process of categorizing resumes and recommending suitable jobs with sophisticated natural language understanding.

## 🚀 Features

### Core Functionality
- **Smart Resume Upload**: Support for PDF, DOC, DOCX, and TXT files
- **AI-Powered Categorization**: Automated candidate classification using machine learning
- **Intelligent Job Matching**: Advanced semantic similarity scoring
- **Interactive Dashboard**: Modern, responsive web interface
- **PostgreSQL Integration**: Robust database management
- **Real-time Analytics**: Comprehensive reporting and insights
- **Secure File Handling**: Safe upload and processing pipeline

### 🤖 Enhanced AI Capabilities
- **BERT-based Understanding**: Uses DistilBERT for semantic comprehension
- **Context-Aware Skill Extraction**: Advanced skill detection with confidence scoring
- **Semantic Job Matching**: Goes beyond keyword matching to understand meaning
- **Transferable Skills Analysis**: Identifies applicable skills across different domains
- **Multi-component Scoring**: Combines TF-IDF, semantic similarity, and skills analysis
- **Experience Level Detection**: Automatically determines candidate seniority
- **Career Transition Support**: Better matching for candidates with non-standard backgrounds

### 🔐 Authentication & User Management
- **Secure Login System**: Session-based authentication with Flask-Login
- **Role-Based Access Control**: Admin and regular user permissions
- **User Management Dashboard**: Admin can create, edit, and manage user accounts
- **Password Security**: Bcrypt hashing with salt for secure password storage
- **Admin Privileges**: Grant/revoke admin access, manage user accounts
- **Session Management**: Secure logout and session handling

### Default Admin Account
- **Email**: `admin@resumeai.com`
- **Password**: `admin123` (change after first login)

## 📋 Prerequisites

Before setting up ResuAI, ensure you have:

- **Python 3.8+** (Recommended: 3.9 or 3.10)
- **PostgreSQL 12+** (Recommended: PostgreSQL 14+)
- **pip** (Python package manager)
- **~2GB free disk space** (for BERT models and dependencies)
- **4GB+ RAM** (for optimal AI model performance)

## 🛠️ Complete Setup Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/Rainieeer/ResuAI.git
cd ResuAI
```

### Step 2: Set Up Python Environment

#### Option A: Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Option B: Using Conda
```bash
conda create -n resumai python=3.9
conda activate resumai
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This will install all required packages including:
- Flask and Flask extensions
- Machine learning libraries (scikit-learn, transformers, torch)
- Database drivers (psycopg2-binary)
- NLP libraries (NLTK, spaCy)
- Data processing libraries (pandas, numpy)

### Step 4: Set Up PostgreSQL Database

#### Install PostgreSQL
- **Windows**: Download from [postgresql.org](https://www.postgresql.org/download/windows/)
- **macOS**: `brew install postgresql` or download from postgresql.org
- **Linux**: `sudo apt-get install postgresql postgresql-contrib`

#### Create Database
```sql
-- Connect to PostgreSQL as superuser
psql -U postgres

-- Create the database
CREATE DATABASE resumai_db;

-- Create a user (optional but recommended)
CREATE USER resumai_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE resumai_db TO resumai_user;

-- Exit psql
\q
```

### Step 5: Configure Environment Variables

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** with your database credentials:
   ```env
   # Database Configuration
   DATABASE_URL=postgresql://postgres:your_password@localhost:5432/resumai_db
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=resumai_db
   DB_USER=postgres
   DB_PASSWORD=your_password

   # Security (IMPORTANT: Change this!)
   SECRET_KEY=your-super-secret-key-change-this-in-production

   # Application Settings
   FLASK_ENV=development
   FLASK_DEBUG=True

   # AI Model Settings
   USE_BERT=True
   SEMANTIC_THRESHOLD=0.7
   BERT_MAX_LENGTH=256
   MAX_EMBEDDING_CACHE=500
   MAX_SIMILARITY_CACHE=1000
   LOG_LEVEL=INFO
   ```

### Step 6: Test Database Connection

```bash
# Test PostgreSQL connection
python -c "import psycopg2; conn = psycopg2.connect('postgresql://postgres:your_password@localhost:5432/resumai_db'); print('✅ Database connection successful!'); conn.close()"
```

### Step 7: Initialize the Database

```bash
# Create database tables and schema
python -c "from database import DatabaseManager; db = DatabaseManager(); print('✅ Database initialized successfully!')"
```

### Step 8: Download AI Models and Dependencies

```bash
# Download NLTK data and test AI models
python semantic_setup.py
```

This will:
- Download required NLTK datasets
- Download BERT and sentence transformer models (~500MB)
- Test semantic analysis functionality
- Verify all AI components are working

### Step 9: Start the Application

```bash
python app.py
```

You should see output similar to:
```
INFO:database:Database initialized successfully
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### Step 10: Access the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

You should see the ResuAI dashboard interface.

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Database Connection Issues
```bash
# Check if PostgreSQL is running
# Windows:
net start postgresql-x64-14

# macOS:
brew services start postgresql

# Linux:
sudo systemctl start postgresql
```

#### Python Package Issues
```bash
# Upgrade pip first
pip install --upgrade pip

# Install packages one by one if batch install fails
pip install Flask Flask-Cors Flask-SQLAlchemy
pip install psycopg2-binary
pip install transformers torch sentence-transformers
```

#### NLTK Download Issues
```python
# Run in Python console if automatic download fails
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

#### Port Already in Use
```bash
# Find process using port 5000
# Windows:
netstat -ano | findstr :5000

# macOS/Linux:
lsof -i :5000

# Kill the process or use a different port
python app.py --port 5001
```

## 📁 Project Structure

```
ResuAI/
├── app.py                  # Main Flask application
├── database.py            # Database management
├── models.py              # SQLAlchemy models
├── utils.py               # Resume processing utilities
├── semantic_setup.py      # AI model setup and testing
├── semantic_demo.py       # Semantic analysis demo
├── requirements.txt       # Python dependencies
├── schema.sql            # Database schema
├── .env                  # Environment variables (create from .env.example)
├── models/               # Pre-trained ML models
│   ├── rf_classifier_categorization.pkl
│   ├── rf_classifier_job_recommendation.pkl
│   ├── tfidf_vectorizer_categorization.pkl
│   └── tfidf_vectorizer_job_recommendation.pkl
├── static/               # Frontend assets
│   ├── css/             # Stylesheets
│   ├── js/              # JavaScript modules
│   └── images/          # Static images
├── templates/           # HTML templates
│   ├── dashboard.html   # Main dashboard
│   ├── index.html      # Landing page
│   └── demo.html       # Demo page
└── temp_uploads/       # Temporary file storage
```

## 🔒 Security & Best Practices

### Environment Security
- **Never commit `.env` files** - they're in `.gitignore` for a reason
- **Use strong passwords** for database connections
- **Change the SECRET_KEY** before deploying to production
- **Regularly update dependencies** to patch security vulnerabilities

### Database Security
```sql
-- Create a dedicated user for the application
CREATE USER resumai_app WITH PASSWORD 'strong_random_password';
GRANT CONNECT ON DATABASE resumai_db TO resumai_app;
GRANT USAGE ON SCHEMA public TO resumai_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO resumai_app;
```

## 🤝 Team Collaboration

### Recommended Setup for Teams

#### Option 1: Local Development (Recommended)
Each developer has their own local setup:
- Individual PostgreSQL databases
- Separate `.env` files
- Independent development environments

#### Option 2: Shared Development Database
Use a cloud database service:
- **Railway**: Easy PostgreSQL hosting
- **Supabase**: Free PostgreSQL with web interface
- **AWS RDS**: Production-grade database hosting
- **Google Cloud SQL**: Scalable database solution

#### Team Workflow
1. **Clone the repository**
2. **Create your own `.env` file** (never commit it)
3. **Set up local database** or use shared development database
4. **Run `semantic_setup.py`** to download AI models
5. **Start development** with `python app.py`

## 🚀 Deployment

### Production Deployment Checklist

1. **Environment Configuration**
   ```env
   FLASK_ENV=production
   FLASK_DEBUG=False
   SECRET_KEY=your-production-secret-key
   DATABASE_URL=postgresql://user:password@your-production-db:5432/resumai_db
   ```

2. **Database Setup**
   - Use a production PostgreSQL instance
   - Enable SSL connections
   - Set up regular backups
   - Configure connection pooling

3. **Web Server**
   - Use Gunicorn or uWSGI instead of Flask's built-in server
   - Set up reverse proxy with Nginx
   - Configure SSL certificates

4. **Monitoring & Logging**
   - Set up application monitoring
   - Configure log aggregation
   - Set up error tracking

### Quick Deploy with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📊 Usage Guide

### First Steps After Setup

1. **Access the Dashboard**: Navigate to `http://localhost:5000`
2. **Create Job Categories**: Set up job categories and requirements
3. **Upload Resumes**: Use the upload interface to add candidate resumes
4. **Review Results**: Check AI-generated match scores and recommendations
5. **Manage Candidates**: Organize and track candidates through the hiring process

### API Endpoints

The application provides REST API endpoints:
- `POST /upload` - Upload resume files
- `GET /candidates` - Retrieve candidate data
- `GET /jobs` - Get job listings
- `GET /analytics` - Analytics data
- `GET /api/candidates/<id>` - Individual candidate details

## 🤖 AI Model Information

### Models Used
- **Sentence Transformers**: `all-MiniLM-L6-v2` for semantic similarity
- **DistilBERT**: `distilbert-base-uncased` for text understanding
- **spaCy**: `en_core_web_sm` for NLP processing
- **Custom TF-IDF**: Trained on resume and job description data
- **Random Forest**: For classification and recommendation

### Model Performance
- **Semantic Similarity**: Threshold of 0.7 for matching
- **Processing Speed**: ~2-3 seconds per resume
- **Memory Usage**: ~1GB for full model loading
- **Accuracy**: 85%+ for job-candidate matching

## 📈 Performance Optimization

### For Better Performance
- **Increase BERT_MAX_LENGTH** for longer documents (up to 512)
- **Adjust cache sizes** based on available memory
- **Use GPU acceleration** for faster model inference
- **Implement caching** for frequently accessed data

### Memory Management
```env
# Adjust these based on your system capabilities
MAX_EMBEDDING_CACHE=1000      # Increase for more caching
MAX_SIMILARITY_CACHE=2000     # Increase for better performance
BERT_MAX_LENGTH=256           # Increase for longer text processing
```

## 🆘 Support

### Getting Help
1. **Check the troubleshooting section** above
2. **Review the error logs** in the terminal
3. **Test individual components** using the provided test scripts
4. **Check GitHub issues** for similar problems

### Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details.

---

**Note**: This is an AI-powered application that requires significant computational resources. For production use, ensure adequate server specifications and consider implementing rate limiting and resource monitoring.
