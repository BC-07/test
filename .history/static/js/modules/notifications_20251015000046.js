/**
 * Comprehensive Notification System for ResuAI
 * Handles toast notifications, notification center, progress indicators, and status updates
 */

class NotificationSystem {
    constructor() {
        this.notifications = [];
        this.toasts = [];
        this.progressItems = [];
        this.isInitialized = false;
        
        // Configuration
        this.config = {
            maxNotifications: 50,
            toastDuration: 5000,
            maxToasts: 5,
            autoMarkAsRead: true,
            persistNotifications: true
        };
        
        this.init();
    }
    
    /**
     * Initialize the notification system
     */
    init() {
        if (this.isInitialized) return;
        
        this.createNotificationCenter();
        this.createToastContainer();
        this.bindEvents();
        this.loadPersistedNotifications();
        this.startStatusMonitoring();
        
        this.isInitialized = true;
        console.log('📢 Notification System initialized');
    }
    
    /**
     * Create the notification center UI
     */
    createNotificationCenter() {
        // Use existing dashboard notification button instead of creating new one
        // Create notification center container (panel only, no button)
        const centerHTML = `
            <div class="notification-center" id="notificationCenter">
                <div class="notification-panel" id="notificationPanel">
                    <div class="notification-header">
                        <h3 class="notification-title">Notifications</h3>
                        <button class="notification-clear" id="notificationClear">Clear All</button>
                    </div>
                    <div class="notification-list" id="notificationList">
                        <div class="notification-empty" id="notificationEmpty">
                            <i class="fas fa-bell-slash" style="font-size: 24px; margin-bottom: 10px; opacity: 0.5;"></i>
                            <p>No notifications yet</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Add to page
        document.body.insertAdjacentHTML('beforeend', centerHTML);
        
        // Add badge to existing dashboard notification button
        this.attachToDashboardButton();
    }
    
    /**
     * Attach notification system to existing dashboard button
     */
    attachToDashboardButton() {
        // Find existing notification button in dashboard - be more specific
        const existingBell = document.querySelector('.top-bar-right .btn-icon[title="Notifications"]');
        
        if (existingBell) {
            // Add badge to existing button
            if (!existingBell.querySelector('.notification-badge')) {
                const badge = document.createElement('div');
                badge.className = 'notification-badge';
                badge.id = 'notificationBadge';
                badge.style.display = 'none';
                badge.textContent = '0';
                existingBell.style.position = 'relative';
                existingBell.appendChild(badge);
            }
            
            console.log('📢 Attached to existing dashboard notification button');
        } else {
            console.warn('Dashboard notification button not found');
        }
    }
    
    /**
     * Create the toast container
     */
    createToastContainer() {
        const containerHTML = `
            <div class="toast-container" id="toastContainer"></div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', containerHTML);
    }
    
    /**
     * Bind event listeners
     */
    bindEvents() {
        // Use existing dashboard notification button
        const existingBell = document.querySelector('.top-bar-right .btn-icon[title="Notifications"], .top-bar-right .btn-icon i.fa-bell')?.closest('.btn-icon');
        const panel = document.getElementById('notificationPanel');
        
        if (existingBell) {
            existingBell.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const isVisible = panel.classList.contains('show');
                if (isVisible) {
                    this.hideNotificationPanel();
                } else {
                    this.showNotificationPanel();
                }
            });
        }
        
        // Clear all notifications
        const clearBtn = document.getElementById('notificationClear');
        clearBtn?.addEventListener('click', () => {
            this.clearAllNotifications();
        });
        
        // Close panel when clicking outside
        document.addEventListener('click', (e) => {
            const center = document.getElementById('notificationCenter');
            if (center && !center.contains(e.target)) {
                this.hideNotificationPanel();
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl+Shift+N - Toggle notifications
            if (e.ctrlKey && e.shiftKey && e.key === 'N') {
                e.preventDefault();
                this.toggleNotificationPanel();
            }
            // Escape - Close panel
            if (e.key === 'Escape') {
                this.hideNotificationPanel();
            }
        });
    }
    
    /**
     * Show notification panel
     */
    showNotificationPanel() {
        const panel = document.getElementById('notificationPanel');
        panel?.classList.add('show');
        
        // Mark all as read
        if (this.config.autoMarkAsRead) {
            this.markAllAsRead();
        }
    }
    
    /**
     * Hide notification panel
     */
    hideNotificationPanel() {
        const panel = document.getElementById('notificationPanel');
        panel?.classList.remove('show');
    }
    
    /**
     * Toggle notification panel
     */
    toggleNotificationPanel() {
        const panel = document.getElementById('notificationPanel');
        if (panel?.classList.contains('show')) {
            this.hideNotificationPanel();
        } else {
            this.showNotificationPanel();
        }
    }
    
    /**
     * Add a new notification
     */
    addNotification(type, title, message, options = {}) {
        const notification = {
            id: this.generateId(),
            type: type, // success, warning, error, info
            title: title,
            message: message,
            timestamp: new Date(),
            read: false,
            persistent: options.persistent !== false,
            action: options.action || null,
            data: options.data || null
        };
        
        // Add to notifications array
        this.notifications.unshift(notification);
        
        // Limit notifications
        if (this.notifications.length > this.config.maxNotifications) {
            this.notifications = this.notifications.slice(0, this.config.maxNotifications);
        }
        
        // Update UI
        this.updateNotificationCenter();
        this.updateBadge();
        
        // Show toast if enabled
        if (options.showToast !== false) {
            this.showToast(type, title, message, options.toastDuration);
        }
        
        // Persist notifications
        if (this.config.persistNotifications) {
            this.persistNotifications();
        }
        
        // Trigger custom event
        this.dispatchNotificationEvent('notification-added', notification);
        
        return notification.id;
    }
    
    /**
     * Show a toast notification
     */
    showToast(type, title, message, duration = null) {
        const toastDuration = duration || this.config.toastDuration;
        const toastId = this.generateId();
        
        // Create toast element
        const toast = this.createToastElement(toastId, type, title, message);
        
        // Add to container
        const container = document.getElementById('toastContainer');
        container?.appendChild(toast);
        
        // Show animation
        setTimeout(() => {
            toast.classList.add('show');
        }, 10);
        
        // Auto remove
        const timeoutId = setTimeout(() => {
            this.removeToast(toastId);
        }, toastDuration);
        
        // Store toast reference
        this.toasts.push({
            id: toastId,
            element: toast,
            timeoutId: timeoutId
        });
        
        // Limit active toasts
        if (this.toasts.length > this.config.maxToasts) {
            const oldestToast = this.toasts.shift();
            this.removeToast(oldestToast.id);
        }
        
        return toastId;
    }
    
    /**
     * Create toast element
     */
    createToastElement(id, type, title, message) {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.dataset.toastId = id;
        
        const iconMap = {
            success: 'fas fa-check',
            warning: 'fas fa-exclamation-triangle',
            error: 'fas fa-times',
            info: 'fas fa-info'
        };
        
        toast.innerHTML = `
            <div class="toast-icon">
                <i class="${iconMap[type] || iconMap.info}"></i>
            </div>
            <div class="toast-content">
                <div class="toast-title">${this.escapeHtml(title)}</div>
                <div class="toast-message">${this.escapeHtml(message)}</div>
            </div>
            <button class="toast-close" title="Close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // Bind close button
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn?.addEventListener('click', () => {
            this.removeToast(id);
        });
        
        return toast;
    }
    
    /**
     * Remove a toast
     */
    removeToast(toastId) {
        const toastIndex = this.toasts.findIndex(t => t.id === toastId);
        if (toastIndex === -1) return;
        
        const toast = this.toasts[toastIndex];
        
        // Clear timeout
        if (toast.timeoutId) {
            clearTimeout(toast.timeoutId);
        }
        
        // Hide animation
        toast.element.classList.add('hide');
        
        // Remove from DOM
        setTimeout(() => {
            toast.element.remove();
        }, 300);
        
        // Remove from array
        this.toasts.splice(toastIndex, 1);
    }
    
    /**
     * Update notification center UI
     */
    updateNotificationCenter() {
        const list = document.getElementById('notificationList');
        const empty = document.getElementById('notificationEmpty');
        
        if (!list) return;
        
        if (this.notifications.length === 0) {
            empty.style.display = 'block';
            list.innerHTML = empty.outerHTML;
            return;
        }
        
        empty.style.display = 'none';
        
        const html = this.notifications.map(notification => {
            return this.createNotificationHTML(notification);
        }).join('');
        
        list.innerHTML = html;
        
        // Bind click events
        list.querySelectorAll('.notification-item').forEach(item => {
            item.addEventListener('click', () => {
                const notificationId = item.dataset.notificationId;
                this.handleNotificationClick(notificationId);
            });
        });
    }
    
    /**
     * Create notification HTML
     */
    createNotificationHTML(notification) {
        const timeAgo = this.formatTimeAgo(notification.timestamp);
        
        return `
            <div class="notification-item ${notification.read ? '' : 'unread'}" 
                 data-notification-id="${notification.id}">
                <div class="notification-meta">
                    <span class="notification-type ${notification.type}">${notification.type}</span>
                    <span class="notification-time">${timeAgo}</span>
                </div>
                <div class="notification-message">
                    <strong>${this.escapeHtml(notification.title)}</strong><br>
                    ${this.escapeHtml(notification.message)}
                </div>
            </div>
        `;
    }
    
    /**
     * Handle notification click
     */
    handleNotificationClick(notificationId) {
        const notification = this.notifications.find(n => n.id === notificationId);
        if (!notification) return;
        
        // Mark as read
        notification.read = true;
        this.updateNotificationCenter();
        this.updateBadge();
        
        // Execute action if available
        if (notification.action) {
            if (typeof notification.action === 'function') {
                notification.action(notification);
            } else if (typeof notification.action === 'string') {
                // Navigate to URL
                window.location.href = notification.action;
            }
        }
        
        // Trigger event
        this.dispatchNotificationEvent('notification-clicked', notification);
    }
    
    /**
     * Update notification badge
     */
    updateBadge() {
        const badge = document.getElementById('notificationBadge');
        if (!badge) return;
        
        const unreadCount = this.notifications.filter(n => !n.read).length;
        
        if (unreadCount > 0) {
            badge.textContent = unreadCount > 99 ? '99+' : unreadCount;
            badge.style.display = 'flex';
        } else {
            badge.style.display = 'none';
        }
    }
    
    /**
     * Mark all notifications as read
     */
    markAllAsRead() {
        this.notifications.forEach(n => n.read = true);
        this.updateNotificationCenter();
        this.updateBadge();
        this.persistNotifications();
    }
    
    /**
     * Clear all notifications
     */
    clearAllNotifications() {
        this.notifications = [];
        this.updateNotificationCenter();
        this.updateBadge();
        this.persistNotifications();
        
        this.showToast('info', 'Cleared', 'All notifications have been cleared');
    }
    
    /**
     * Show progress notification
     */
    showProgress(title, options = {}) {
        const progressId = this.generateId();
        const progress = {
            id: progressId,
            title: title,
            percentage: options.percentage || 0,
            status: options.status || 'Starting...',
            indeterminate: options.indeterminate || false,
            persistent: true
        };
        
        this.progressItems.push(progress);
        this.createProgressElement(progress);
        
        return progressId;
    }
    
    /**
     * Update progress
     */
    updateProgress(progressId, updates = {}) {
        const progress = this.progressItems.find(p => p.id === progressId);
        if (!progress) return;
        
        Object.assign(progress, updates);
        this.updateProgressElement(progress);
    }
    
    /**
     * Complete progress
     */
    completeProgress(progressId, message = 'Completed successfully') {
        const progress = this.progressItems.find(p => p.id === progressId);
        if (!progress) return;
        
        // Show completion toast
        this.showToast('success', progress.title, message);
        
        // Remove progress
        this.removeProgress(progressId);
    }
    
    /**
     * Remove progress
     */
    removeProgress(progressId) {
        const progressIndex = this.progressItems.findIndex(p => p.id === progressId);
        if (progressIndex === -1) return;
        
        // Remove from array
        this.progressItems.splice(progressIndex, 1);
        
        // Remove element
        const element = document.querySelector(`[data-progress-id="${progressId}"]`);
        if (element) {
            element.style.opacity = '0';
            setTimeout(() => element.remove(), 300);
        }
    }
    
    /**
     * Create progress element
     */
    createProgressElement(progress) {
        const container = document.getElementById('toastContainer');
        if (!container) return;
        
        const element = document.createElement('div');
        element.className = 'progress-notification';
        element.dataset.progressId = progress.id;
        element.innerHTML = this.createProgressHTML(progress);
        
        container.appendChild(element);
    }
    
    /**
     * Create progress HTML
     */
    createProgressHTML(progress) {
        return `
            <div class="progress-header">
                <div class="progress-title">${this.escapeHtml(progress.title)}</div>
                <div class="progress-percentage">${progress.indeterminate ? '' : progress.percentage + '%'}</div>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar ${progress.indeterminate ? 'indeterminate' : ''}" 
                     style="width: ${progress.indeterminate ? '100%' : progress.percentage + '%'}"></div>
            </div>
            <div class="progress-status">${this.escapeHtml(progress.status)}</div>
        `;
    }
    
    /**
     * Update progress element
     */
    updateProgressElement(progress) {
        const element = document.querySelector(`[data-progress-id="${progress.id}"]`);
        if (!element) return;
        
        element.innerHTML = this.createProgressHTML(progress);
    }
    
    /**
     * Show loading overlay
     */
    showLoading(message = 'Loading...') {
        let overlay = document.getElementById('loadingOverlay');
        
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'loadingOverlay';
            overlay.className = 'loading-overlay';
            overlay.innerHTML = `
                <div class="loading-content">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">${this.escapeHtml(message)}</div>
                </div>
            `;
            document.body.appendChild(overlay);
        } else {
            overlay.querySelector('.loading-text').textContent = message;
        }
        
        setTimeout(() => overlay.classList.add('show'), 10);
    }
    
    /**
     * Hide loading overlay
     */
    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.remove('show');
        }
    }
    
    /**
     * Start system status monitoring
     */
    startStatusMonitoring() {
        // Monitor system status every 30 seconds
        setInterval(() => {
            this.checkSystemStatus();
        }, 30000);
        
        // Initial check
        this.checkSystemStatus();
    }
    
    /**
     * Check system status
     */
    async checkSystemStatus() {
        try {
            const response = await fetch('/api/system/status');
            const status = await response.json();
            
            if (status.success) {
                this.updateSystemStatus(status.data);
            }
        } catch (error) {
            console.warn('Failed to check system status:', error);
        }
    }
    
    /**
     * Update system status indicators
     */
    updateSystemStatus(status) {
        // Update status indicators throughout the UI
        const indicators = document.querySelectorAll('.status-indicator[data-status-type]');
        
        indicators.forEach(indicator => {
            const type = indicator.dataset.statusType;
            if (status[type] !== undefined) {
                this.updateStatusIndicator(indicator, status[type]);
            }
        });
        
        // Check for important status changes
        if (status.database === false) {
            this.addNotification('error', 'Database Error', 'Database connection lost', {
                persistent: true,
                action: () => window.location.reload()
            });
        }
    }
    
    /**
     * Update status indicator
     */
    updateStatusIndicator(indicator, status) {
        const statusClass = status ? 'online' : 'offline';
        const statusText = status ? 'Online' : 'Offline';
        
        indicator.className = `status-indicator ${statusClass}`;
        indicator.innerHTML = `
            <div class="status-dot ${status ? 'pulse' : ''}"></div>
            ${statusText}
        `;
    }
    
    /**
     * Persist notifications to localStorage
     */
    persistNotifications() {
        try {
            const data = {
                notifications: this.notifications.slice(0, 20), // Keep last 20
                timestamp: Date.now()
            };
            localStorage.setItem('resumeai_notifications', JSON.stringify(data));
        } catch (error) {
            console.warn('Failed to persist notifications:', error);
        }
    }
    
    /**
     * Load persisted notifications
     */
    loadPersistedNotifications() {
        try {
            const data = localStorage.getItem('resumeai_notifications');
            if (data) {
                const parsed = JSON.parse(data);
                // Only load if less than 24 hours old
                if (Date.now() - parsed.timestamp < 24 * 60 * 60 * 1000) {
                    this.notifications = parsed.notifications.map(n => ({
                        ...n,
                        timestamp: new Date(n.timestamp)
                    }));
                    this.updateNotificationCenter();
                    this.updateBadge();
                }
            }
        } catch (error) {
            console.warn('Failed to load persisted notifications:', error);
        }
    }
    
    /**
     * Utility methods
     */
    generateId() {
        return 'notif_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    formatTimeAgo(timestamp) {
        const now = new Date();
        const diff = now - timestamp;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);
        
        if (minutes < 1) return 'Just now';
        if (minutes < 60) return `${minutes}m ago`;
        if (hours < 24) return `${hours}h ago`;
        return `${days}d ago`;
    }
    
    dispatchNotificationEvent(type, data) {
        const event = new CustomEvent(type, { detail: data });
        document.dispatchEvent(event);
    }
    
    /**
     * Public API shortcuts
     */
    success(title, message, options = {}) {
        return this.addNotification('success', title, message, options);
    }
    
    error(title, message, options = {}) {
        return this.addNotification('error', title, message, options);
    }
    
    warning(title, message, options = {}) {
        return this.addNotification('warning', title, message, options);
    }
    
    info(title, message, options = {}) {
        return this.addNotification('info', title, message, options);
    }
}

// Initialize global notification system
let notifications;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        notifications = new NotificationSystem();
        window.notifications = notifications; // Make globally available
    });
} else {
    notifications = new NotificationSystem();
    window.notifications = notifications;
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NotificationSystem;
}