/**
 * Enhanced Loading Utility
 * Provides beautiful loading animations and states
 */

class LoadingUtils {
    constructor() {
        this.overlay = null;
        this.currentLoadingId = null;
    }

    /**
     * Show a loading overlay
     * @param {Object} options - Loading configuration
     * @param {string} options.type - Loading type: 'robot', 'spinner', 'dots', 'progress'
     * @param {string} options.message - Loading message
     * @param {string} options.subtext - Additional loading text
     * @param {boolean} options.dismissible - Whether clicking overlay dismisses it
     * @returns {string} Loading ID for management
     */
    show(options = {}) {
        const config = {
            type: 'robot',
            message: 'Processing...',
            subtext: 'Please wait while we work our magic',
            dismissible: false,
            ...options
        };

        // Remove existing overlay
        this.hide();

        // Generate unique ID
        this.currentLoadingId = `loading-${Date.now()}`;

        // Create overlay
        this.overlay = document.createElement('div');
        this.overlay.className = 'loading-overlay';
        this.overlay.setAttribute('data-loading-id', this.currentLoadingId);

        // Create loading content
        const container = document.createElement('div');
        container.className = 'loading-container';
        
        container.innerHTML = this.getLoadingHTML(config);

        // Handle dismissible overlay
        if (config.dismissible) {
            this.overlay.addEventListener('click', (e) => {
                if (e.target === this.overlay) {
                    this.hide();
                }
            });
        }

        this.overlay.appendChild(container);
        document.body.appendChild(this.overlay);

        // Show with animation
        requestAnimationFrame(() => {
            this.overlay.classList.add('active');
        });

        return this.currentLoadingId;
    }

    /**
     * Hide the loading overlay
     */
    hide() {
        if (this.overlay) {
            this.overlay.classList.remove('active');
            
            setTimeout(() => {
                if (this.overlay && this.overlay.parentNode) {
                    this.overlay.parentNode.removeChild(this.overlay);
                }
                this.overlay = null;
                this.currentLoadingId = null;
            }, 300);
        }
    }

    /**
     * Update loading message
     */
    updateMessage(message, subtext = null) {
        if (this.overlay) {
            const messageEl = this.overlay.querySelector('.loading-text');
            const subtextEl = this.overlay.querySelector('.loading-subtext');
            
            if (messageEl) messageEl.textContent = message;
            if (subtextEl && subtext) subtextEl.textContent = subtext;
        }
    }

    /**
     * Get loading HTML based on type
     */
    getLoadingHTML(config) {
        const animations = {
            robot: this.getRobotHTML(),
            spinner: this.getSpinnerHTML(),
            dots: this.getDotsHTML(),
            progress: this.getProgressHTML()
        };

        const animationHTML = animations[config.type] || animations.robot;

        return `
            ${animationHTML}
            <div class="loading-text">${this.escapeHtml(config.message)}</div>
            <div class="loading-subtext">${this.escapeHtml(config.subtext)}</div>
        `;
    }

    getRobotHTML() {
        return `
            <div class="loading-robot">
                <div class="robot-head">
                    <div class="robot-antenna"></div>
                    <div class="robot-arms"></div>
                </div>
                <div class="robot-body"></div>
            </div>
        `;
    }

    getSpinnerHTML() {
        return `
            <div class="loading-spinner">
                <div class="spinner-circle"></div>
            </div>
        `;
    }

    getDotsHTML() {
        return `
            <div class="spinner-dots">
                <div class="spinner-dot"></div>
                <div class="spinner-dot"></div>
                <div class="spinner-dot"></div>
            </div>
        `;
    }

    getProgressHTML() {
        return `
            <div class="loading-spinner">
                <div class="spinner-circle"></div>
            </div>
            <div class="loading-progress">
                <div class="loading-progress-bar"></div>
            </div>
        `;
    }

    /**
     * Show button loading state
     */
    showButtonLoading(button, text = null) {
        if (!button) return;
        
        // Store original content
        if (!button.hasAttribute('data-original-html')) {
            button.setAttribute('data-original-html', button.innerHTML);
        }
        
        button.classList.add('btn-loading');
        button.disabled = true;
        
        if (text) {
            button.innerHTML = `<span class="btn-text">${text}</span>`;
        }
    }

    /**
     * Hide button loading state
     */
    hideButtonLoading(button) {
        if (!button) return;
        
        button.classList.remove('btn-loading');
        button.disabled = false;
        
        const originalHtml = button.getAttribute('data-original-html');
        if (originalHtml) {
            button.innerHTML = originalHtml;
            button.removeAttribute('data-original-html');
        }
    }

    /**
     * Show card loading state
     */
    showCardLoading(card) {
        if (card) {
            card.classList.add('card-loading');
        }
    }

    /**
     * Hide card loading state
     */
    hideCardLoading(card) {
        if (card) {
            card.classList.remove('card-loading');
        }
    }

    /**
     * Create skeleton loader
     */
    createSkeleton(container, type = 'text', count = 3) {
        if (!container) return;
        
        container.innerHTML = '';
        
        for (let i = 0; i < count; i++) {
            const skeleton = document.createElement('div');
            skeleton.className = `skeleton skeleton-${type}`;
            container.appendChild(skeleton);
        }
    }

    /**
     * Quick loading methods
     */
    showAIProcessing() {
        return this.show({
            type: 'robot',
            message: 'AI Processing',
            subtext: 'Analyzing resumes with artificial intelligence...'
        });
    }

    showUploading() {
        return this.show({
            type: 'progress',
            message: 'Uploading Files',
            subtext: 'Please wait while we upload your resumes...'
        });
    }

    showGeneratingReport() {
        return this.show({
            type: 'spinner',
            message: 'Generating Report',
            subtext: 'Compiling your analytics data...'
        });
    }

    showSaving() {
        return this.show({
            type: 'dots',
            message: 'Saving Changes',
            subtext: 'Your data is being saved...'
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Create global instance
const loadingUtils = new LoadingUtils();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LoadingUtils;
}

// Global access
window.LoadingUtils = loadingUtils;
window.showLoading = (options) => loadingUtils.show(options);
window.hideLoading = () => loadingUtils.hide();
window.updateLoadingMessage = (message, subtext) => loadingUtils.updateMessage(message, subtext);
