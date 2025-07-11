/**
 * Enhanced Confirmation Dialog Utility
 * Replaces basic browser confirm() with beautiful, customizable dialogs
 */

class ConfirmationDialog {
    constructor() {
        this.overlay = null;
        this.currentResolve = null;
        this.currentReject = null;
    }

    /**
     * Show a confirmation dialog
     * @param {Object} options - Dialog configuration
     * @param {string} options.title - Dialog title
     * @param {string} options.message - Main message
     * @param {string} options.details - Additional details (optional)
     * @param {string} options.confirmText - Confirm button text (default: 'Confirm')
     * @param {string} options.cancelText - Cancel button text (default: 'Cancel')
     * @param {string} options.type - Dialog type: 'danger', 'success', 'warning', 'info' (default: 'danger')
     * @param {string} options.icon - Custom icon (default: based on type)
     * @param {boolean} options.dangerousAction - Whether to emphasize this is a dangerous action
     * @returns {Promise<boolean>} - Resolves to true if confirmed, false if cancelled
     */
    show(options = {}) {
        return new Promise((resolve, reject) => {
            // Store resolve/reject for button handlers
            this.currentResolve = resolve;
            this.currentReject = reject;

            // Default options
            const config = {
                title: 'Confirm Action',
                message: 'Are you sure you want to proceed?',
                details: '',
                confirmText: 'Confirm',
                cancelText: 'Cancel',
                type: 'danger',
                icon: null,
                dangerousAction: false,
                ...options
            };

            // Create and show dialog
            this.createDialog(config);
            this.showDialog();
        });
    }

    /**
     * Quick method for deletion confirmations
     */
    confirmDelete(itemName = 'this item') {
        return this.show({
            title: 'Delete Confirmation',
            message: `Are you sure you want to delete ${itemName}?`,
            details: 'This action cannot be undone.',
            confirmText: 'Delete',
            cancelText: 'Keep',
            type: 'danger',
            icon: 'fas fa-trash-alt',
            dangerousAction: true
        });
    }

    /**
     * Quick method for removal confirmations
     */
    confirmRemove(itemName = 'this item') {
        return this.show({
            title: 'Remove Confirmation',
            message: `Are you sure you want to remove ${itemName}?`,
            details: 'You can always add it back later.',
            confirmText: 'Remove',
            cancelText: 'Keep',
            type: 'warning',
            icon: 'fas fa-user-times'
        });
    }

    /**
     * Quick method for save confirmations
     */
    confirmSave(hasChanges = true) {
        if (!hasChanges) return Promise.resolve(true);
        
        return this.show({
            title: 'Save Changes',
            message: 'Do you want to save your changes?',
            details: 'Your changes will be lost if you don\'t save them.',
            confirmText: 'Save',
            cancelText: 'Don\'t Save',
            type: 'info',
            icon: 'fas fa-save'
        });
    }

    createDialog(config) {
        // Remove existing dialog
        this.removeDialog();

        // Create overlay
        this.overlay = document.createElement('div');
        this.overlay.className = 'confirmation-dialog-overlay';

        // Create dialog
        const dialog = document.createElement('div');
        dialog.className = `confirmation-dialog ${config.type}`;

        // Get icon based on type
        const icon = config.icon || this.getDefaultIcon(config.type);

        // Build dialog HTML
        dialog.innerHTML = `
            <div class="confirmation-dialog-header">
                <div class="confirmation-dialog-icon">
                    <i class="${icon}"></i>
                </div>
                <h3 class="confirmation-dialog-title">${this.escapeHtml(config.title)}</h3>
            </div>
            <div class="confirmation-dialog-body">
                <p class="confirmation-dialog-message">${this.escapeHtml(config.message)}</p>
                ${config.details ? `<p class="confirmation-dialog-details">${this.escapeHtml(config.details)}</p>` : ''}
            </div>
            <div class="confirmation-dialog-actions">
                <button type="button" class="confirmation-dialog-btn confirmation-dialog-btn-cancel">
                    <i class="fas fa-times"></i>
                    <span>${this.escapeHtml(config.cancelText)}</span>
                </button>
                <button type="button" class="confirmation-dialog-btn confirmation-dialog-btn-confirm">
                    <i class="${this.getConfirmIcon(config.type)}"></i>
                    <span>${this.escapeHtml(config.confirmText)}</span>
                </button>
            </div>
        `;

        // Append to overlay
        this.overlay.appendChild(dialog);

        // Add event listeners
        this.setupEventListeners(dialog);

        // Append to body
        document.body.appendChild(this.overlay);
    }

    setupEventListeners(dialog) {
        const cancelBtn = dialog.querySelector('.confirmation-dialog-btn-cancel');
        const confirmBtn = dialog.querySelector('.confirmation-dialog-btn-confirm');

        // Cancel button
        cancelBtn.addEventListener('click', () => {
            this.handleCancel();
        });

        // Confirm button
        confirmBtn.addEventListener('click', () => {
            this.handleConfirm();
        });

        // Overlay click (cancel)
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) {
                this.handleCancel();
            }
        });

        // Escape key
        this.escapeKeyHandler = (e) => {
            if (e.key === 'Escape') {
                this.handleCancel();
            }
        };
        document.addEventListener('keydown', this.escapeKeyHandler);

        // Enter key (confirm)
        this.enterKeyHandler = (e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.altKey) {
                this.handleConfirm();
            }
        };
        document.addEventListener('keydown', this.enterKeyHandler);
    }

    showDialog() {
        // Force reflow to ensure styles are applied
        this.overlay.offsetHeight;
        
        // Add active class for animation
        this.overlay.classList.add('active');
        
        // Focus the cancel button by default for accessibility
        setTimeout(() => {
            const cancelBtn = this.overlay.querySelector('.confirmation-dialog-btn-cancel');
            if (cancelBtn) {
                cancelBtn.focus();
            }
        }, 100);
    }

    hideDialog() {
        return new Promise((resolve) => {
            if (!this.overlay) {
                resolve();
                return;
            }

            // Remove active class for exit animation
            this.overlay.classList.remove('active');

            // Wait for animation to complete
            setTimeout(() => {
                this.removeDialog();
                resolve();
            }, 300);
        });
    }

    removeDialog() {
        if (this.overlay) {
            // Remove event listeners
            if (this.escapeKeyHandler) {
                document.removeEventListener('keydown', this.escapeKeyHandler);
                this.escapeKeyHandler = null;
            }
            if (this.enterKeyHandler) {
                document.removeEventListener('keydown', this.enterKeyHandler);
                this.enterKeyHandler = null;
            }

            // Remove from DOM
            this.overlay.remove();
            this.overlay = null;
        }
    }

    handleConfirm() {
        this.hideDialog().then(() => {
            if (this.currentResolve) {
                this.currentResolve(true);
                this.currentResolve = null;
                this.currentReject = null;
            }
        });
    }

    handleCancel() {
        this.hideDialog().then(() => {
            if (this.currentResolve) {
                this.currentResolve(false);
                this.currentResolve = null;
                this.currentReject = null;
            }
        });
    }

    getDefaultIcon(type) {
        switch (type) {
            case 'danger': return 'fas fa-exclamation-triangle';
            case 'success': return 'fas fa-check-circle';
            case 'warning': return 'fas fa-exclamation-circle';
            case 'info': return 'fas fa-info-circle';
            default: return 'fas fa-question-circle';
        }
    }

    getConfirmIcon(type) {
        switch (type) {
            case 'danger': return 'fas fa-trash-alt';
            case 'success': return 'fas fa-check';
            case 'warning': return 'fas fa-exclamation';
            case 'info': return 'fas fa-arrow-right';
            default: return 'fas fa-check';
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Create global instance
const confirmDialog = new ConfirmationDialog();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConfirmationDialog;
}

// Global methods for easy access
window.confirmDialog = confirmDialog;

// Convenience methods
window.showConfirm = (options) => confirmDialog.show(options);
window.confirmDelete = (itemName) => confirmDialog.confirmDelete(itemName);
window.confirmRemove = (itemName) => confirmDialog.confirmRemove(itemName);
window.confirmSave = (hasChanges) => confirmDialog.confirmSave(hasChanges);
