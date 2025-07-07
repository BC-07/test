// Local Storage Service
const StorageService = {
    // Set item in localStorage with error handling
    setItem(key, value) {
        try {
            const serializedValue = typeof value === 'string' ? value : JSON.stringify(value);
            localStorage.setItem(key, serializedValue);
            return true;
        } catch (error) {
            console.error(`Failed to save to localStorage:`, error);
            return false;
        }
    },

    // Get item from localStorage with error handling
    getItem(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            if (item === null) return defaultValue;
            
            // Try to parse as JSON, fallback to string
            try {
                return JSON.parse(item);
            } catch {
                return item;
            }
        } catch (error) {
            console.error(`Failed to read from localStorage:`, error);
            return defaultValue;
        }
    },

    // Remove item from localStorage
    removeItem(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (error) {
            console.error(`Failed to remove from localStorage:`, error);
            return false;
        }
    },

    // Clear all localStorage
    clear() {
        try {
            localStorage.clear();
            return true;
        } catch (error) {
            console.error(`Failed to clear localStorage:`, error);
            return false;
        }
    },

    // App-specific methods
    app: {
        // Theme management
        getTheme() {
            return StorageService.getItem('theme', 'light');
        },

        setTheme(theme) {
            return StorageService.setItem('theme', theme);
        },

        // Sidebar state
        getSidebarCollapsed() {
            return StorageService.getItem('sidebarCollapsed', false);
        },

        setSidebarCollapsed(collapsed) {
            return StorageService.setItem('sidebarCollapsed', collapsed);
        },

        // User preferences
        getUserPreferences() {
            return StorageService.getItem('userPreferences', {});
        },

        setUserPreferences(preferences) {
            return StorageService.setItem('userPreferences', preferences);
        }
    }
};

// Make available globally
window.StorageService = StorageService;
