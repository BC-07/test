// Bootstrap Components Initialization
const BootstrapInit = {
    // Initialize all Bootstrap components
    init() {
        this.initTooltips();
        this.initModals();
        this.initPopovers();
    },

    // Initialize tooltips
    initTooltips() {
        try {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
            console.log(`Initialized ${tooltipTriggerList.length} tooltips`);
        } catch (error) {
            console.error('Failed to initialize tooltips:', error);
        }
    },

    // Initialize modals
    initModals() {
        try {
            const modalElements = document.querySelectorAll('.modal');
            modalElements.forEach(modalElement => {
                try {
                    new bootstrap.Modal(modalElement);
                } catch (error) {
                    console.error(`Failed to initialize modal ${modalElement.id}:`, error);
                }
            });
            console.log(`Initialized ${modalElements.length} modals`);
        } catch (error) {
            console.error('Failed to initialize modals:', error);
        }
    },

    // Initialize popovers
    initPopovers() {
        try {
            const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
            popoverTriggerList.map(function (popoverTriggerEl) {
                return new bootstrap.Popover(popoverTriggerEl);
            });
            console.log(`Initialized ${popoverTriggerList.length} popovers`);
        } catch (error) {
            console.error('Failed to initialize popovers:', error);
        }
    },

    // Get modal instance
    getModal(modalId) {
        try {
            const modalElement = document.getElementById(modalId);
            if (modalElement) {
                return bootstrap.Modal.getInstance(modalElement) || new bootstrap.Modal(modalElement);
            }
            return null;
        } catch (error) {
            console.error(`Failed to get modal instance for ${modalId}:`, error);
            return null;
        }
    },

    // Show modal
    showModal(modalId) {
        const modal = this.getModal(modalId);
        if (modal) {
            modal.show();
        } else {
            console.error(`Modal with ID '${modalId}' not found`);
        }
    },

    // Hide modal
    hideModal(modalId) {
        const modal = this.getModal(modalId);
        if (modal) {
            modal.hide();
        } else {
            console.error(`Modal with ID '${modalId}' not found`);
        }
    }
};

// Make available globally
window.BootstrapInit = BootstrapInit;

// Keep backward compatibility
window.initializeBootstrapComponents = BootstrapInit.init.bind(BootstrapInit);
