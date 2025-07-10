# ResumeAI

ResumeAI is an advanced resume screening and job recommendation web application powered by machine learning and BERT-based semantic analysis. It helps automate the process of categorizing resumes and recommending suitable jobs with sophisticated natural language understanding.

## Features

### Core Functionality
- Upload and screen resumes (PDF, DOC, DOCX, TXT)
- Categorize candidates using machine learning models
- Job recommendation system with comprehensive match scoring
- Modern, user-friendly dashboard
- PostgreSQL database integration
- Secure file handling
- Real-time analytics and reporting

### 🆕 Enhanced Semantic Analysis
- **BERT-based Understanding**: Uses DistilBERT for semantic comprehension
- **Advanced Skill Extraction**: Context-aware skill detection with confidence scoring
- **Semantic Job Matching**: Goes beyond keyword matching to understand meaning
- **Transferable Skills Analysis**: Identifies applicable skills across different domains
- **Multi-component Scoring**: Combines TF-IDF, semantic similarity, and skills analysis
- **Experience Level Detection**: Automatically determines candidate seniority
- **Career Transition Support**: Better matching for candidates with non-standard backgrounds

## Prerequisites

- Python 3.8+
- PostgreSQL 12+
- pip (Python package manager)
- ~500MB additional storage for BERT models

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ResuAI.git
   cd ResuAI
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up PostgreSQL:**
   - Create a PostgreSQL database named `resumai_db`
   - Note your database credentials (username, password, host, port)

5. **Configure environment variables:**
   - Copy `.env.example` to `.env`
   - Update the database connection details in `.env`:
   ```
   DATABASE_URL=postgresql://username:password@localhost:5432/resumai_db
   SECRET_KEY=your-secret-key-here
   ```

6. **Initialize the database:**
   ```bash
   python -c "from database import DatabaseManager; DatabaseManager()"
   ```

7. **Run the application:**
   ```bash
   python app.py
   ```

## Database Setup

The application uses PostgreSQL. Make sure you have:

1. **PostgreSQL installed and running**
2. **Database created:**
   ```sql
   CREATE DATABASE resumai_db;
   ```
3. **Environment variables configured** (see `.env.example`)

## Usage

1. **Access the application:** Open `http://localhost:5000` in your browser
2. **Create job categories and jobs** in the dashboard
3. **Upload resumes** and let the AI analyze them
4. **Review candidates** with match scores and recommendations
5. **Manage the hiring process** through the dashboard

## Team Collaboration

When working with teammates, you have several options:

### Option 1: Each Developer Has Their Own Local Database
**Best for: Development and testing**
- Each team member installs PostgreSQL locally
- Everyone creates their own `resumai_db` database
- Each person uses their own `.env` file with local credentials
- Data is not shared between team members

### Option 2: Shared Cloud Database
**Best for: Production and team coordination**
- Use a cloud PostgreSQL service (AWS RDS, Google Cloud SQL, Railway, Supabase)
- All team members share the same database connection
- Everyone uses the same DATABASE_URL in their `.env` file
- ⚠️ **Be careful**: All team members will share the same data

### Option 3: Network-Accessible Database
**Best for: Local team on same network**
- One team member hosts the PostgreSQL database
- Configure PostgreSQL to accept remote connections
- Other team members connect to the host's IP address
- ⚠️ **Security risk**: Not recommended for production

**Each team member needs:**
- Their own `.env` file with appropriate database credentials
- The same Python environment (use `requirements.txt`)
- Access to the chosen database (local or shared)

## Security Notes

- Never commit `.env` files to version control
- Use strong passwords for database connections
- Keep your `SECRET_KEY` secure
- Regularly update dependencies

## Project Structure

```
app.py                  # Main application entry point (Flask app)
database.py             # Database setup and utilities
models.py               # SQLAlchemy models
utils.py                # Utility functions for resume processing
requirements.txt        # Python dependencies
schema.sql              # Database schema
static/                 # CSS, JS, and images for the frontend
templates/              # HTML templates (dashboard, landing page)
models/                 # Machine learning model files
temp_uploads/           # Temporary file uploads
.env.example            # Environment variables template
```

## Quick Setup for Teammates

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ResuAI.git
   cd ResuAI
   ```

2. **Install PostgreSQL locally:**
   - Download from https://www.postgresql.org/download/
   - Create a database: `CREATE DATABASE resumai_db;`

3. **Set up the project:**
   ```bash
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your database credentials
   python -c "from database import DatabaseManager; DatabaseManager()"
   python app.py
   ```

## Important Notes for GitHub

- ✅ The `.env` file is in `.gitignore` - your database credentials are safe
- ✅ Each team member needs their own `.env` file
- ✅ Your PostgreSQL database is only accessible from your machine by default
- ❌ **Your coworkers cannot connect to your local database** - they need their own setup

## License

MIT License
