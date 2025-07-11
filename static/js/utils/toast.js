// Enhanced Toast Notification System
const ToastUtils = {
    toastCounter: 0,
    
    // Create a dynamic toast element
    createToast(message, type = 'success', options = {}) {
        const defaults = {
            duration: 5000,
            dismissible: true,
            showIcon: true,
            title: null
        };
        
        const config = { ...defaults, ...options };
        const toastId = `toast-${++this.toastCounter}`;
        
        // Get or create toast container
        let container = document.querySelector('.toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
        
        // Create toast element
        const toast = document.createElement('div');
        toast.id = toastId;
        toast.className = `toast toast-${type}`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        // Get icon and title based on type
        const { icon, title } = this.getTypeInfo(type, config.title);
        
        toast.innerHTML = `
            <div class="toast-header">
                ${config.showIcon ? `<i class="${icon}"></i>` : ''}
                <strong>${title}</strong>
                ${config.dismissible ? '<button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"><i class="fas fa-times"></i></button>' : ''}
            </div>
            <div class="toast-body">
                <div class="toast-content">
                    <div class="toast-message">${this.escapeHtml(message)}</div>
                </div>
            </div>
        `;
        
        // Add to container
        container.appendChild(toast);
        
        // Initialize Bootstrap toast
        const bsToast = new bootstrap.Toast(toast, {
            delay: config.duration,
            autohide: config.duration > 0
        });
        
        // Show toast
        bsToast.show();
        
        // Auto-remove after hiding
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
        
        return bsToast;
    },
    
    // Show toast notification (enhanced version)
    showToast(message, type = 'success', options = {}) {
        // Try new dynamic method first
        try {
            return this.createToast(message, type, options);
        } catch (error) {
            console.error('Failed to create dynamic toast, falling back:', error);
            
            // Fallback to existing static toasts
            const toast = document.getElementById(`${type}Toast`);
            if (toast) {
                const toastBody = toast.querySelector('.toast-body');
                if (toastBody) {
                    // Check if toast body has toast-content structure
                    const toastContent = toastBody.querySelector('.toast-content');
                    const messageElement = toastContent ? 
                        toastContent.querySelector('.toast-message') || toastContent :
                        toastBody;
                    
                    messageElement.textContent = message;
                }
                
                try {
                    const bsToast = new bootstrap.Toast(toast);
                    bsToast.show();
                    return bsToast;
                } catch (error) {
                    console.error('Failed to show static toast:', error);
                }
            }
            
            // Final fallback
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    },

    // Get icon and title based on toast type
    getTypeInfo(type, customTitle) {
        const typeMap = {
            success: { icon: 'fas fa-check-circle', title: 'Success' },
            error: { icon: 'fas fa-times-circle', title: 'Error' },
            warning: { icon: 'fas fa-exclamation-circle', title: 'Warning' },
            info: { icon: 'fas fa-info-circle', title: 'Info' }
        };
        
        const info = typeMap[type] || typeMap.info;
        return {
            icon: info.icon,
            title: customTitle || info.title
        };
    },
    
    // Escape HTML to prevent XSS
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    // Show success toast
    showSuccess(message, options = {}) {
        return this.showToast(message, 'success', options);
    },

    // Show error toast
    showError(message, options = {}) {
        return this.showToast(message, 'error', options);
    },

    // Show warning toast
    showWarning(message, options = {}) {
        return this.showToast(message, 'warning', options);
    },

    // Show info toast
    showInfo(message, options = {}) {
        return this.showToast(message, 'info', options);
    },
    
    // Show custom toast with more options
    showCustom(message, type, title, duration = 5000) {
        return this.showToast(message, type, { title, duration });
    },
    
    // Show persistent toast (doesn't auto-hide)
    showPersistent(message, type = 'info', title = null) {
        return this.showToast(message, type, { 
            title, 
            duration: 0, // 0 means no auto-hide
            dismissible: true 
        });
    }
};

// Make available globally
window.ToastUtils = ToastUtils;

// Keep backward compatibility
window.showToast = ToastUtils.showToast.bind(ToastUtils);
