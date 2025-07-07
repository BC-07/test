# ResumeAI

ResumeAI is a resume screening and job recommendation web application powered by machine learning. It helps automate the process of categorizing resumes and recommending suitable jobs.

## Features

- Upload and screen resumes
- Categorize candidates using machine learning models
- Job recommendation system
- Modern, user-friendly dashboard
- Secure file handling

## Project Structure

```
app.py                  # Main application entry point (Flask app)
database.py             # Database setup and utilities
models.py               # Model loading and prediction logic
requirements.txt        # Python dependencies
static/                 # CSS, JS, and images for the frontend
templates/              # HTML templates (dashboard, landing page)
utils.py                # Utility functions
resume_screening.db     # SQLite database
models/                 # (Ignored) Machine learning model files
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Rainieeer/ResuAI.git
   cd ResuAI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and go to `http://localhost:5000`

## Notes

- The `models/` directory is ignored in git due to large files. Place your trained model files there if needed.
- For production, configure environment variables and secure your database.

## License

MIT License
