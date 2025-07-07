// Theme Management Component
const ThemeManager = {
    // Initialize theme functionality
    init() {
        this.themeToggle = DOMUtils.getElementById('themeToggle');
        this.html = document.documentElement;
        
        this.loadSavedTheme();
        this.setupEventListeners();
    },

    // Load saved theme from storage
    loadSavedTheme() {
        const savedTheme = StorageService.app.getTheme();
        this.setTheme(savedTheme);
    },

    // Set theme
    setTheme(theme) {
        if (!this.html) return;
        
        this.html.setAttribute('data-theme', theme);
        
        if (this.themeToggle) {
            this.themeToggle.innerHTML = theme === 'dark' 
                ? '<i class="fas fa-sun"></i>' 
                : '<i class="fas fa-moon"></i>';
        }
        
        // Save theme preference
        StorageService.app.setTheme(theme);
        
        console.log(`Theme set to: ${theme}`);
    },

    // Toggle theme
    toggleTheme() {
        const currentTheme = this.html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);
    },

    // Setup event listeners
    setupEventListeners() {
        if (this.themeToggle) {
            this.themeToggle.addEventListener('click', () => {
                this.toggleTheme();
            });
        }
    },

    // Get current theme
    getCurrentTheme() {
        return this.html.getAttribute('data-theme') || 'light';
    }
};

// Make available globally
window.ThemeManager = ThemeManager;

// Keep backward compatibility
window.setupThemeToggle = ThemeManager.init.bind(ThemeManager);
