// Application Configuration
const CONFIG = {
    // API Endpoints
    API: {
        UPLOAD: '/api/upload',
        JOBS: '/api/jobs',
        CANDIDATES: '/api/candidates',
        ANALYTICS: '/api/analytics',
        SETTINGS: '/api/settings'
    },
    
    // File upload settings
    UPLOAD: {
        MAX_FILE_SIZE: 16 * 1024 * 1024, // 16MB
        ALLOWED_TYPES: ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'],
        ALLOWED_EXTENSIONS: ['.pdf', '.docx', '.txt']
    },
    
    // UI Settings
    UI: {
        TOAST_DURATION: 5000,
        ANIMATION_DURATION: 300
    }
};

// Make config globally available
window.CONFIG = CONFIG;
