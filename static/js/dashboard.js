// Main Dashboard Application
// All utility functions and services are now modularized
// This file now focuses on orchestrating the main application functionality

// Keep backward compatibility for existing functions
const API = CONFIG.API;
const escapeHtml = DOMUtils.escapeHtml;
const showToast = ToastUtils.showToast;
const formatDate = FormatUtils.formatDate;
const formatFileSize = FormatUtils.formatFileSize;

// Navigation Setup - now delegated to NavigationModule
function setupNavigation() {
    // Deprecated: Use NavigationModule.init() instead
    NavigationModule.init();
}

// Sidebar toggle
function setupSidebarToggle() {
    const sidebarToggle = document.getElementById('sidebarToggle');
    const floatingSidebarToggle = document.getElementById('floatingSidebarToggle');
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('mainContent');
    
    // Load saved state from storage
    const sidebarCollapsed = StorageService.app.getSidebarCollapsed();
    if (sidebarCollapsed) {
        sidebar.classList.add('collapsed');
        mainContent.classList.add('expanded');
    }
    
    // Toggle function
    const toggleSidebar = () => {
        sidebar.classList.toggle('collapsed');
        mainContent.classList.toggle('expanded');
        StorageService.app.setSidebarCollapsed(sidebar.classList.contains('collapsed'));
    };
    
    // Attach click handlers
    if (sidebarToggle && sidebar && mainContent) {
        sidebarToggle.addEventListener('click', toggleSidebar);
    }
    
    if (floatingSidebarToggle && sidebar && mainContent) {
        floatingSidebarToggle.addEventListener('click', toggleSidebar);
    }
    
    // Add keyboard shortcut (Ctrl+B or Cmd+B) to toggle sidebar
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
            e.preventDefault();
            window.toggleSidebar();
        }
    });
}

// Global function to show sidebar (accessible from console)
window.showSidebar = function() {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('mainContent');
    
    if (sidebar && mainContent) {
        sidebar.classList.remove('collapsed');
        mainContent.classList.remove('expanded');
        StorageService.app.setSidebarCollapsed(false);
        console.log('Sidebar is now visible');
    }
};

// Global function to toggle sidebar (accessible from console)
window.toggleSidebar = function() {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('mainContent');
    
    if (sidebar && mainContent) {
        sidebar.classList.toggle('collapsed');
        mainContent.classList.toggle('expanded');
        StorageService.app.setSidebarCollapsed(sidebar.classList.contains('collapsed'));
        console.log('Sidebar toggled:', sidebar.classList.contains('collapsed') ? 'hidden' : 'visible');
    }
};

// Initialize all functionality
document.addEventListener('DOMContentLoaded', () => {
    // Initialize modular components first
    BootstrapInit.init();
    ThemeManager.init();
    
    // Initialize feature modules
    NavigationModule.init();
    UploadModule.init();
    JobsModule.init();
    CandidatesModule.init();
    AnalyticsModule.init();
    DashboardModule.init();
    
    // Setup application features
    setupSidebarToggle();
    
    // Note: Dashboard data loading is now handled by DashboardModule.init()
});

// Keep backward compatibility - delegate to new components
function initializeBootstrapComponents() {
    BootstrapInit.init();
}

function setupThemeToggle() {
    // Deprecated: Use ThemeManager.init() instead
    ThemeManager.init();
}

// Resume Upload Global Variables - now managed by UploadModule
let selectedJobId = null;
let selectedFiles = [];

// Resume Upload Functionality - now delegated to UploadModule
function setupResumeUpload() {
    UploadModule.init();
    
    // Sync global variables for backward compatibility
    Object.defineProperty(window, 'selectedJobId', {
        get: () => UploadModule.selectedJobId,
        set: (value) => { UploadModule.selectedJobId = value; }
    });
    
    Object.defineProperty(window, 'selectedFiles', {
        get: () => UploadModule.selectedFiles,
        set: (value) => { UploadModule.selectedFiles = value; }
    });
}

// Export function to be used by job selection - now delegated to UploadModule
function selectJobForUpload(jobId) {
    UploadModule.selectJob(jobId);
}

// Export function globally
window.selectJobForUpload = selectJobForUpload;

// Clear job selection function - now delegated to UploadModule
window.clearJobSelection = function() {
    UploadModule.selectedJobId = null;
    
    // Remove selected class from all job cards
    document.querySelectorAll('.job-category-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    // Hide selected job details
    const selectedJobDetails = document.getElementById('selectedJobDetails');
    if (selectedJobDetails) selectedJobDetails.style.display = 'none';
    
    // Hide upload zone and show instructions
    const uploadZone = document.getElementById('uploadZone');
    const uploadInstructions = document.getElementById('uploadInstructions');
    if (uploadZone) uploadZone.style.display = 'none';
    if (uploadInstructions) uploadInstructions.style.display = 'block';
    
    showToast('Job selection cleared', 'info');
};

// Helper functions - now delegated to UploadModule
function updateUploadButtonState() {
    UploadModule.updateUploadButtonState();
}

function updateFileStats() {
    UploadModule.updateFileStats();
}

function clearSelectedFiles() {
    UploadModule.clearSelectedFiles();
}

// Upload functionality - now delegated to UploadModule
async function loadJobCategoriesForUpload() {
    // Deprecated: Use UploadModule.loadJobCategories() instead
    UploadModule.loadJobCategories();
}

function displayRankingResults(results) {
    // Deprecated: Use UploadModule.displayRankingResults() instead
    UploadModule.displayRankingResults(results);
}

function getScoreColorClass(score) {
    // Deprecated: Use DOMUtils.getScoreColorClass() instead
    return DOMUtils.getScoreColorClass(score);
}

// Job Management Functionality - now delegated to JobsModule
function setupJobManagement() {
    // Deprecated: Use JobsModule.init() instead
    JobsModule.init();
}

// Analytics Functionality
// Analytics Functionality - now delegated to AnalyticsModule
async function loadAnalytics() {
    // Deprecated: Use AnalyticsModule.loadAnalytics() instead
    AnalyticsModule.loadAnalytics();
}

function updateMetricsCards(summary) {
    // Deprecated: Use AnalyticsModule.updateMetricsCards() instead
    AnalyticsModule.updateMetricsCards(summary);
}

function updateMetricsTable(dailyStats) {
    // Deprecated: Use AnalyticsModule.updateMetricsTable() instead
    AnalyticsModule.updateMetricsTable(dailyStats);
}

function updateCharts(dailyStats) {
    // Deprecated: Use AnalyticsModule.updateCharts() instead
    AnalyticsModule.updateCharts(dailyStats);
}

// Candidates Section Loading - now delegated to CandidatesModule
async function loadCandidatesSection() {
    // Deprecated: Use CandidatesModule.loadCandidates() instead
    CandidatesModule.loadCandidates();
}

function displayCandidatesByJob(candidatesByJob, totalCandidates) {
    // Deprecated: Use CandidatesModule.displayCandidatesByJob() instead
    CandidatesModule.displayCandidatesByJob(candidatesByJob, totalCandidates);
}

function setupCandidateActionListeners() {
    // Deprecated: Use CandidatesModule.setupActionListeners() instead
    CandidatesModule.setupActionListeners();
}

async function showCandidateDetails(candidateId, modal) {
    // Deprecated: Use CandidatesModule.showCandidateDetails() instead
    CandidatesModule.showCandidateDetails(candidateId, modal);
}

async function removeCandidate(candidateId) {
    // Deprecated: Use CandidatesModule.removeCandidate() instead
    CandidatesModule.removeCandidate(candidateId);
}

async function updateCandidateStatus(candidateId, status) {
    // Deprecated: Use CandidatesModule.updateCandidateStatus() instead
    CandidatesModule.updateCandidateStatus(candidateId, status);
}

// Dashboard Data Loading
// Dashboard Data Loading - now delegated to DashboardModule
async function loadDashboardData() {
    // Deprecated: Use DashboardModule.loadDashboardData() instead
    DashboardModule.loadDashboardData();
}

// Refresh all data sections after updates - now delegated to DashboardModule
async function refreshAllData() {
    // Deprecated: Use DashboardModule.refreshAllData() instead
    DashboardModule.refreshAllData();
}

// Load candidates data for the candidates section - now delegated to CandidatesModule
async function loadCandidates() {
    // Deprecated: Use CandidatesModule.loadCandidates() instead
    CandidatesModule.loadCandidates();
}

// =============================================================================
// REFACTORING COMPLETE
// =============================================================================
// This file has been successfully refactored and modularized.
// All major functionality has been delegated to appropriate modules:
//
// - NavigationModule: Navigation and routing
// - UploadModule: Resume upload and job selection
// - JobsModule: Job management and categories
// - CandidatesModule: Candidate management and display
// - AnalyticsModule: Analytics, charts, and metrics
// - DashboardModule: Dashboard data loading and widgets
//
// Legacy functions are maintained for backward compatibility but are
// deprecated and delegate to their respective modules.
// =============================================================================