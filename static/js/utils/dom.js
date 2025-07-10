// DOM Utilities
const DOMUtils = {
    // Safe HTML escaping
    escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') return '';
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    },

    // Get element by ID with error handling
    getElementById(id) {
        const element = document.getElementById(id);
        if (!element) {
            console.warn(`Element with ID '${id}' not found`);
        }
        return element;
    },

    // Get elements by selector with error handling
    querySelectorAll(selector) {
        try {
            return document.querySelectorAll(selector);
        } catch (error) {
            console.error(`Invalid selector: ${selector}`, error);
            return [];
        }
    },

    // Show/hide element
    toggleElement(element, show) {
        if (!element) return;
        element.style.display = show ? 'block' : 'none';
    },

    // Add/remove CSS classes
    toggleClass(element, className, add) {
        if (!element) return;
        if (add) {
            element.classList.add(className);
        } else {
            element.classList.remove(className);
        }
    },

    // Create element with attributes and content
    createElement(tag, attributes = {}, content = '') {
        const element = document.createElement(tag);
        
        Object.entries(attributes).forEach(([key, value]) => {
            element.setAttribute(key, value);
        });
        
        if (content) {
            element.innerHTML = content;
        }
        
        return element;
    },

    // Get CSS class for score ranges
    getScoreColorClass(score) {
        if (score >= 90) return 'score-excellent';
        if (score >= 75) return 'score-good';
        if (score >= 60) return 'score-fair';
        return 'score-poor';
    }
};

// Make available globally
window.DOMUtils = DOMUtils;
