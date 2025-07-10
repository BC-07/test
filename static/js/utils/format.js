// Formatting Utilities
const FormatUtils = {
    // Format date to readable string
    formatDate(dateString) {
        if (!dateString) return 'N/A';
        try {
            return new Date(dateString).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        } catch (error) {
            console.error('Invalid date:', dateString);
            return 'Invalid Date';
        }
    },

    // Format file size to human readable format
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        if (typeof bytes !== 'number' || bytes < 0) return 'Invalid Size';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        if (i >= sizes.length) return 'Too Large';
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    // Format percentage
    formatPercentage(value, decimals = 0) {
        if (typeof value !== 'number') return '0%';
        return `${value.toFixed(decimals)}%`;
    },

    // Truncate text with ellipsis
    truncateText(text, maxLength = 100) {
        if (!text || typeof text !== 'string') return '';
        return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    },

    // Format skill tags HTML
    formatSkillTags(skills, maxDisplay = 5, className = 'skill-tag') {
        if (!Array.isArray(skills)) return '';
        
        const displaySkills = skills.slice(0, maxDisplay);
        let html = displaySkills.map(skill => 
            `<span class="${className}">${DOMUtils.escapeHtml(skill)}</span>`
        ).join('');
        
        if (skills.length > maxDisplay) {
            html += `<span class="${className} bg-secondary">+${skills.length - maxDisplay}</span>`;
        }
        
        return html;
    }
};

// Make available globally
window.FormatUtils = FormatUtils;
