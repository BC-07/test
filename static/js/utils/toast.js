// Toast Notification System
const ToastUtils = {
    // Show toast notification
    showToast(message, type = 'success') {
        const toast = document.getElementById(`${type}Toast`);
        if (toast) {
            const toastBody = toast.querySelector('.toast-body');
            if (toastBody) {
                toastBody.textContent = message;
            }
            
            try {
                const bsToast = new bootstrap.Toast(toast);
                bsToast.show();
            } catch (error) {
                console.error('Failed to show toast:', error);
                // Fallback to console log
                console.log(`${type.toUpperCase()}: ${message}`);
            }
        } else {
            console.warn(`Toast element '${type}Toast' not found`);
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    },

    // Show success toast
    showSuccess(message) {
        this.showToast(message, 'success');
    },

    // Show error toast
    showError(message) {
        this.showToast(message, 'error');
    },

    // Show warning toast
    showWarning(message) {
        this.showToast(message, 'warning');
    },

    // Show info toast
    showInfo(message) {
        this.showToast(message, 'info');
    }
};

// Make available globally
window.ToastUtils = ToastUtils;

// Keep backward compatibility
window.showToast = ToastUtils.showToast.bind(ToastUtils);
