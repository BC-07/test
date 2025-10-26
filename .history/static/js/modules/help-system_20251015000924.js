/**
 * Interactive Help & Question System for ResuAI
 * Provides contextual help, FAQ, and guided assistance
 */

class HelpSystem {
    constructor() {
        this.isInitialized = false;
        this.currentSection = null;
        this.searchIndex = [];
        this.contextualHelp = new Map();
        
        // Help content database
        this.helpContent = {
            getting_started: {
                title: "Getting Started",
                icon: "fas fa-play-circle",
                items: [
                    {
                        title: "How to upload a PDS file",
                        description: "Learn how to upload and process Personal Data Sheet files",
                        content: "Navigate to the 'Upload PDS' section and drag & drop your Excel file or click to browse. The system supports .xlsx and .xls formats.",
                        action: () => this.highlightElement('[data-help="upload-pds"]')
                    },
                    {
                        title: "Understanding the dashboard",
                        description: "Overview of the main dashboard features",
                        content: "The dashboard shows analytics, recent candidates, and system status. Use the navigation menu to access different sections.",
                        action: () => this.highlightElement('[data-help="dashboard"]')
                    },
                    {
                        title: "Viewing candidate assessments",
                        description: "How to access and interpret candidate evaluations",
                        content: "Click on any candidate to view their detailed assessment including scores, qualifications, and recommendations.",
                        action: () => this.highlightElement('[data-help="candidates"]')
                    }
                ]
            },
            assessment: {
                title: "Assessment System",
                icon: "fas fa-chart-line",
                items: [
                    {
                        title: "How assessments work",
                        description: "Understanding the university assessment criteria",
                        content: "Our system evaluates candidates based on Education (40%), Experience (20%), Training (10%), Eligibility (10%), Accomplishments (5%), and Potential (15%).",
                        action: () => this.showAssessmentGuide()
                    },
                    {
                        title: "Viewing assessment results",
                        description: "How to interpret candidate scores and rankings",
                        content: "Scores range from 0-100. Green indicates excellent (80+), yellow is good (60-79), and red needs improvement (<60).",
                        action: () => this.highlightElement('[data-help="assessment-results"]')
                    },
                    {
                        title: "Customizing assessment criteria",
                        description: "Adjusting evaluation parameters for different positions",
                        content: "Assessment criteria can be customized per job posting to match specific requirements and preferences.",
                        action: () => this.showCriteriaEditor()
                    }
                ]
            },
            job_management: {
                title: "Job Management",
                icon: "fas fa-briefcase",
                items: [
                    {
                        title: "Creating job postings",
                        description: "How to add new university positions",
                        content: "Use the Job Management section to create detailed job postings with requirements, qualifications, and assessment criteria.",
                        action: () => this.navigateToJobManagement()
                    },
                    {
                        title: "Managing applicants",
                        description: "Organizing and reviewing job applications",
                        content: "View all applicants for each position, filter by qualifications, and track application status.",
                        action: () => this.highlightElement('[data-help="applicants"]')
                    },
                    {
                        title: "Generating reports",
                        description: "Creating hiring reports and analytics",
                        content: "Generate comprehensive reports on hiring metrics, candidate quality, and position fulfillment rates.",
                        action: () => this.showReportingGuide()
                    }
                ]
            },
            analytics: {
                title: "Analytics & Reports",
                icon: "fas fa-chart-bar",
                items: [
                    {
                        title: "Understanding analytics dashboard",
                        description: "How to read the analytics visualizations",
                        content: "The analytics dashboard shows real-time metrics on candidate processing, assessment scores, and hiring trends.",
                        action: () => this.highlightElement('[data-help="analytics"]')
                    },
                    {
                        title: "Exporting data",
                        description: "How to export candidate and assessment data",
                        content: "Use the export features to download candidate lists, assessment results, and analytical reports in various formats.",
                        action: () => this.showExportOptions()
                    },
                    {
                        title: "Setting up automated reports",
                        description: "Configure regular reporting schedules",
                        content: "Set up automated email reports for hiring metrics, candidate summaries, and system status updates.",
                        action: () => this.showAutomationSettings()
                    }
                ]
            },
            troubleshooting: {
                title: "Troubleshooting",
                icon: "fas fa-tools",
                items: [
                    {
                        title: "File upload issues",
                        description: "Common problems with PDS file uploads",
                        content: "Ensure files are in Excel format (.xlsx/.xls), under 10MB, and contain valid PDS data structure.",
                        action: () => this.showUploadTroubleshooting()
                    },
                    {
                        title: "Assessment errors",
                        description: "Resolving assessment calculation problems",
                        content: "Assessment errors usually occur due to incomplete PDS data. Check that all required fields are properly filled.",
                        action: () => this.showAssessmentTroubleshooting()
                    },
                    {
                        title: "System performance",
                        description: "Improving system speed and reliability",
                        content: "Clear browser cache, ensure stable internet connection, and contact support for persistent issues.",
                        action: () => this.showPerformanceTips()
                    }
                ]
            }
        };
        
        this.init();
    }
    
    /**
     * Initialize the help system
     */
    init() {
        if (this.isInitialized) return;
        
        this.attachToDashboardButton();
        this.createHelpPanel();
        this.bindEvents();
        this.buildSearchIndex();
        this.setupContextualHelp();
        this.initializeKeyboardShortcuts();
        
        this.isInitialized = true;
        console.log('❓ Help System initialized');
    }
    
    /**
     * Attach to existing dashboard help button instead of creating floating button
     */
    attachToDashboardButton() {
        // Find existing help button in dashboard - be more specific
        const existingHelp = document.querySelector('.top-bar-right .btn-icon[title="Help"]');
        
        if (existingHelp) {
            console.log('❓ Attached to existing dashboard help button');
        } else {
            console.warn('Dashboard help button not found');
        }
    }
    
    /**
     * Create the help panel
     */
    createHelpPanel() {
        const panelHTML = `
            <div class="help-panel" id="helpPanel">
                <div class="help-panel-header">
                    <div class="help-panel-title">
                        <i class="fas fa-question-circle"></i>
                        Help & Support
                    </div>
                    <div class="help-panel-subtitle">Get help with ResuAI features</div>
                </div>
                
                <div class="help-search">
                    <div class="help-search-container">
                        <input type="text" class="help-search-input" id="helpSearchInput" 
                               placeholder="Search help articles...">
                        <i class="fas fa-search help-search-icon"></i>
                    </div>
                    <div class="help-search-results" id="helpSearchResults"></div>
                </div>
                
                <div class="help-panel-content">
                    <div class="help-sections" id="helpSections">
                        ${this.createHelpSectionsHTML()}
                    </div>
                    
                    <div class="help-quick-actions">
                        <h4>Quick Actions</h4>
                        <div class="quick-action-buttons">
                            <button class="quick-action-btn" id="helpTour">
                                <i class="fas fa-route"></i>
                                Take Tour
                            </button>
                            <button class="quick-action-btn" id="helpContact">
                                <i class="fas fa-envelope"></i>
                                Contact Support
                            </button>
                            <button class="quick-action-btn" id="helpKeyboard">
                                <i class="fas fa-keyboard"></i>
                                Shortcuts
                            </button>
                            <button class="quick-action-btn" id="helpReport">
                                <i class="fas fa-bug"></i>
                                Report Issue
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', panelHTML);
    }
    
    /**
     * Create help sections HTML
     */
    createHelpSectionsHTML() {
        return Object.entries(this.helpContent).map(([key, section]) => {
            return `
                <div class="help-section" data-section="${key}">
                    <div class="help-section-header">
                        <div class="help-section-title">
                            <i class="${section.icon} help-section-icon"></i>
                            ${section.title}
                        </div>
                        <i class="fas fa-chevron-down help-section-chevron"></i>
                    </div>
                    <div class="help-section-content">
                        ${section.items.map(item => `
                            <div class="help-item" data-item="${this.slugify(item.title)}">
                                <div class="help-item-title">${item.title}</div>
                                <div class="help-item-description">${item.description}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }).join('');
    }
    
    /**
     * Bind event listeners
     */
    bindEvents() {
        // Use existing dashboard help button - be more specific
        const existingHelp = document.querySelector('.top-bar-right .btn-icon[title="Help"]');
        
        if (existingHelp) {
            existingHelp.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.toggleHelpPanel();
            });
            console.log('❓ Event listener attached to help button');
        } else {
            console.warn('Could not find help button to attach events');
        }
        
        // Section expansion
        document.addEventListener('click', (e) => {
            const sectionHeader = e.target.closest('.help-section-header');
            if (sectionHeader) {
                const section = sectionHeader.closest('.help-section');
                this.toggleSection(section);
            }
        });
        
        // Help item clicks
        document.addEventListener('click', (e) => {
            const helpItem = e.target.closest('.help-item');
            if (helpItem) {
                this.handleHelpItemClick(helpItem);
            }
        });
        
        // Search functionality
        const searchInput = document.getElementById('helpSearchInput');
        searchInput?.addEventListener('input', (e) => {
            this.handleSearch(e.target.value);
        });
        
        // Quick actions
        document.getElementById('helpTour')?.addEventListener('click', () => {
            this.startGuidedTour();
        });
        
        document.getElementById('helpContact')?.addEventListener('click', () => {
            this.showContactForm();
        });
        
        document.getElementById('helpKeyboard')?.addEventListener('click', () => {
            this.showKeyboardShortcuts();
        });
        
        document.getElementById('helpReport')?.addEventListener('click', () => {
            this.showBugReportForm();
        });
        
        // Close panel when clicking outside
        document.addEventListener('click', (e) => {
            const helpPanel = document.getElementById('helpPanel');
            const helpButton = document.getElementById('helpButton');
            
            if (helpPanel && helpButton && 
                !helpPanel.contains(e.target) && 
                !helpButton.contains(e.target)) {
                this.hideHelpPanel();
            }
        });
    }
    
    /**
     * Toggle help panel visibility
     */
    toggleHelpPanel() {
        const panel = document.getElementById('helpPanel');
        if (panel?.classList.contains('show')) {
            this.hideHelpPanel();
        } else {
            this.showHelpPanel();
        }
    }
    
    /**
     * Show help panel
     */
    showHelpPanel() {
        const panel = document.getElementById('helpPanel');
        const button = document.getElementById('helpButton');
        
        panel?.classList.add('show');
        button?.classList.remove('pulsing');
        
        // Track usage
        this.trackHelpUsage('panel_opened');
    }
    
    /**
     * Hide help panel
     */
    hideHelpPanel() {
        const panel = document.getElementById('helpPanel');
        panel?.classList.remove('show');
    }
    
    /**
     * Toggle section expansion
     */
    toggleSection(section) {
        if (!section) return;
        
        const isExpanded = section.classList.contains('expanded');
        
        // Close all other sections
        document.querySelectorAll('.help-section.expanded').forEach(s => {
            if (s !== section) {
                s.classList.remove('expanded');
            }
        });
        
        // Toggle current section
        section.classList.toggle('expanded', !isExpanded);
        
        if (!isExpanded) {
            this.currentSection = section.dataset.section;
            this.trackHelpUsage('section_opened', this.currentSection);
        }
    }
    
    /**
     * Handle help item click
     */
    handleHelpItemClick(item) {
        const itemId = item.dataset.item;
        const sectionKey = item.closest('.help-section').dataset.section;
        const section = this.helpContent[sectionKey];
        
        if (section) {
            const helpItem = section.items.find(i => this.slugify(i.title) === itemId);
            if (helpItem) {
                this.showHelpItemDetail(helpItem);
                this.trackHelpUsage('item_clicked', `${sectionKey}.${itemId}`);
            }
        }
    }
    
    /**
     * Show help item detail
     */
    showHelpItemDetail(item) {
        // Show detailed help content
        if (window.notifications) {
            window.notifications.info(item.title, item.content, {
                persistent: true,
                toastDuration: 8000
            });
        }
        
        // Execute action if available
        if (item.action && typeof item.action === 'function') {
            setTimeout(() => {
                item.action();
            }, 500);
        }
    }
    
    /**
     * Build search index
     */
    buildSearchIndex() {
        this.searchIndex = [];
        
        Object.entries(this.helpContent).forEach(([sectionKey, section]) => {
            section.items.forEach(item => {
                this.searchIndex.push({
                    section: sectionKey,
                    sectionTitle: section.title,
                    title: item.title,
                    description: item.description,
                    content: item.content,
                    keywords: [
                        ...item.title.toLowerCase().split(' '),
                        ...item.description.toLowerCase().split(' '),
                        ...item.content.toLowerCase().split(' ')
                    ].filter(word => word.length > 2),
                    item: item
                });
            });
        });
    }
    
    /**
     * Handle search
     */
    handleSearch(query) {
        const searchResults = document.getElementById('helpSearchResults');
        const sections = document.getElementById('helpSections');
        
        if (!query.trim()) {
            searchResults.classList.remove('show');
            sections.style.display = 'block';
            return;
        }
        
        const results = this.searchHelpContent(query);
        this.displaySearchResults(results);
        
        if (results.length > 0) {
            searchResults.classList.add('show');
            sections.style.display = 'none';
        } else {
            searchResults.classList.remove('show');
            sections.style.display = 'block';
        }
        
        this.trackHelpUsage('search', query);
    }
    
    /**
     * Search help content
     */
    searchHelpContent(query) {
        const searchTerms = query.toLowerCase().split(' ').filter(term => term.length > 1);
        
        return this.searchIndex.filter(item => {
            return searchTerms.some(term => 
                item.keywords.some(keyword => keyword.includes(term)) ||
                item.title.toLowerCase().includes(term) ||
                item.description.toLowerCase().includes(term)
            );
        }).slice(0, 8); // Limit results
    }
    
    /**
     * Display search results
     */
    displaySearchResults(results) {
        const container = document.getElementById('helpSearchResults');
        if (!container) return;
        
        if (results.length === 0) {
            container.innerHTML = `
                <div class="help-search-result">
                    <div class="help-search-result-title">No results found</div>
                    <div class="help-search-result-snippet">Try different keywords or browse the sections below</div>
                </div>
            `;
            return;
        }
        
        container.innerHTML = results.map(result => `
            <div class="help-search-result" data-section="${result.section}" data-item="${this.slugify(result.title)}">
                <div class="help-search-result-title">${this.highlightSearchTerms(result.title, document.getElementById('helpSearchInput').value)}</div>
                <div class="help-search-result-snippet">${this.truncateText(result.description, 100)}</div>
            </div>
        `).join('');
        
        // Bind click events to results
        container.querySelectorAll('.help-search-result').forEach(result => {
            result.addEventListener('click', () => {
                const section = result.dataset.section;
                const item = result.dataset.item;
                
                if (section && item) {
                    this.openHelpItem(section, item);
                }
            });
        });
    }
    
    /**
     * Open specific help item
     */
    openHelpItem(sectionKey, itemId) {
        // Clear search
        const searchInput = document.getElementById('helpSearchInput');
        if (searchInput) searchInput.value = '';
        this.handleSearch('');
        
        // Open section
        const section = document.querySelector(`[data-section="${sectionKey}"]`);
        if (section) {
            section.classList.add('expanded');
            
            // Find and highlight item
            const item = section.querySelector(`[data-item="${itemId}"]`);
            if (item) {
                setTimeout(() => {
                    item.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    item.style.background = 'rgba(59, 130, 246, 0.1)';
                    setTimeout(() => {
                        item.style.background = '';
                    }, 2000);
                }, 300);
            }
        }
    }
    
    /**
     * Start guided tour
     */
    startGuidedTour() {
        this.hideHelpPanel();
        
        const tourSteps = [
            {
                element: '[data-help="dashboard"]',
                title: 'Dashboard Overview',
                content: 'This is your main dashboard showing analytics and system status.'
            },
            {
                element: '[data-help="upload-pds"]',
                title: 'Upload PDS Files',
                content: 'Upload Personal Data Sheet files here for candidate assessment.'
            },
            {
                element: '[data-help="candidates"]',
                title: 'Candidate Management',
                content: 'View and manage all candidate profiles and assessments.'
            },
            {
                element: '[data-help="analytics"]',
                title: 'Analytics & Reports',
                content: 'Access detailed analytics and generate reports here.'
            }
        ];
        
        this.runGuidedTour(tourSteps);
        this.trackHelpUsage('tour_started');
    }
    
    /**
     * Run guided tour
     */
    runGuidedTour(steps) {
        let currentStep = 0;
        
        const showStep = () => {
            if (currentStep >= steps.length) {
                this.endGuidedTour();
                return;
            }
            
            const step = steps[currentStep];
            const element = document.querySelector(step.element);
            
            if (element) {
                this.highlightElement(step.element);
                this.showTooltip(element, step.title, step.content, () => {
                    currentStep++;
                    setTimeout(showStep, 500);
                });
            } else {
                currentStep++;
                showStep();
            }
        };
        
        showStep();
    }
    
    /**
     * End guided tour
     */
    endGuidedTour() {
        this.clearHighlights();
        
        if (window.notifications) {
            window.notifications.success('Tour Complete', 'You\'ve completed the guided tour! Feel free to explore or ask for help anytime.');
        }
        
        this.trackHelpUsage('tour_completed');
    }
    
    /**
     * Highlight element
     */
    highlightElement(selector) {
        this.clearHighlights();
        
        const element = document.querySelector(selector);
        if (element) {
            element.classList.add('help-highlight');
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
    
    /**
     * Clear highlights
     */
    clearHighlights() {
        document.querySelectorAll('.help-highlight').forEach(el => {
            el.classList.remove('help-highlight');
        });
    }
    
    /**
     * Show tooltip
     */
    showTooltip(element, title, content, onNext) {
        const tooltip = document.createElement('div');
        tooltip.className = 'contextual-help show';
        tooltip.innerHTML = `
            <div style="font-weight: bold; margin-bottom: 5px;">${title}</div>
            <div>${content}</div>
            <div style="margin-top: 8px; text-align: right;">
                <button onclick="this.parentElement.parentElement.remove(); ${onNext ? 'arguments[0]()' : ''}" 
                        style="background: white; color: #1f2937; border: none; padding: 3px 8px; border-radius: 3px; cursor: pointer; font-size: 11px;">
                    Next
                </button>
            </div>
        `;
        
        // Position tooltip
        const rect = element.getBoundingClientRect();
        tooltip.style.top = (rect.bottom + 10) + 'px';
        tooltip.style.left = (rect.left + rect.width / 2) + 'px';
        tooltip.style.transform = 'translateX(-50%)';
        
        document.body.appendChild(tooltip);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (tooltip.parentElement) {
                tooltip.remove();
            }
        }, 10000);
    }
    
    /**
     * Setup contextual help
     */
    setupContextualHelp() {
        // Add help attributes to elements
        document.querySelectorAll('[data-help]').forEach(element => {
            element.addEventListener('mouseenter', (e) => {
                this.showContextualHelp(e.target);
            });
            
            element.addEventListener('mouseleave', () => {
                this.hideContextualHelp();
            });
        });
    }
    
    /**
     * Show contextual help
     */
    showContextualHelp(element) {
        const helpKey = element.dataset.help;
        const helpText = this.contextualHelp.get(helpKey);
        
        if (helpText) {
            this.showTooltip(element, 'Help', helpText);
        }
    }
    
    /**
     * Hide contextual help
     */
    hideContextualHelp() {
        const existingTooltip = document.querySelector('.contextual-help');
        if (existingTooltip) {
            existingTooltip.remove();
        }
    }
    
    /**
     * Initialize keyboard shortcuts
     */
    initializeKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // F1 - Toggle help
            if (e.key === 'F1') {
                e.preventDefault();
                this.toggleHelpPanel();
            }
            
            // Ctrl+Shift+H - Toggle help
            if (e.ctrlKey && e.shiftKey && e.key === 'H') {
                e.preventDefault();
                this.toggleHelpPanel();
            }
            
            // Ctrl+Shift+T - Start tour
            if (e.ctrlKey && e.shiftKey && e.key === 'T') {
                e.preventDefault();
                this.startGuidedTour();
            }
        });
    }
    
    /**
     * Show keyboard shortcuts
     */
    showKeyboardShortcuts() {
        const shortcuts = [
            { key: 'F1', description: 'Toggle help panel' },
            { key: 'Ctrl+Shift+H', description: 'Toggle help panel' },
            { key: 'Ctrl+Shift+T', description: 'Start guided tour' },
            { key: 'Ctrl+Shift+N', description: 'Toggle notifications' },
            { key: 'Escape', description: 'Close panels' }
        ];
        
        const content = shortcuts.map(s => 
            `<div style="display: flex; justify-content: space-between; margin: 5px 0;">
                <code style="background: #f3f4f6; padding: 2px 6px; border-radius: 3px;">${s.key}</code>
                <span>${s.description}</span>
            </div>`
        ).join('');
        
        if (window.notifications) {
            window.notifications.info('Keyboard Shortcuts', content, {
                persistent: true,
                toastDuration: 10000
            });
        }
    }
    
    /**
     * Track help usage
     */
    trackHelpUsage(action, details = '') {
        // Analytics tracking
        try {
            console.log(`Help Usage: ${action}${details ? ' - ' + details : ''}`);
            
            // Could integrate with analytics service
            if (window.gtag) {
                gtag('event', 'help_usage', {
                    action: action,
                    details: details
                });
            }
        } catch (error) {
            // Ignore tracking errors
        }
    }
    
    /**
     * Utility methods
     */
    slugify(text) {
        return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    }
    
    highlightSearchTerms(text, query) {
        if (!query) return text;
        
        const terms = query.split(' ').filter(term => term.length > 1);
        let result = text;
        
        terms.forEach(term => {
            const regex = new RegExp(`(${term})`, 'gi');
            result = result.replace(regex, '<mark>$1</mark>');
        });
        
        return result;
    }
    
    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substr(0, maxLength) + '...';
    }
    
    /**
     * Feature-specific help methods
     */
    showAssessmentGuide() {
        // Implementation for assessment guide
        this.navigateToSection('assessment');
    }
    
    showCriteriaEditor() {
        // Implementation for criteria editor
        this.highlightElement('[data-help="criteria-editor"]');
    }
    
    navigateToJobManagement() {
        // Implementation for job management navigation
        window.location.href = '#job-management';
    }
    
    showReportingGuide() {
        // Implementation for reporting guide
        this.highlightElement('[data-help="reports"]');
    }
    
    showExportOptions() {
        // Implementation for export options
        this.highlightElement('[data-help="export"]');
    }
    
    showAutomationSettings() {
        // Implementation for automation settings
        this.highlightElement('[data-help="automation"]');
    }
    
    showUploadTroubleshooting() {
        // Implementation for upload troubleshooting
        this.openHelpItem('troubleshooting', 'file-upload-issues');
    }
    
    showAssessmentTroubleshooting() {
        // Implementation for assessment troubleshooting
        this.openHelpItem('troubleshooting', 'assessment-errors');
    }
    
    showPerformanceTips() {
        // Implementation for performance tips
        this.openHelpItem('troubleshooting', 'system-performance');
    }
    
    showContactForm() {
        // Implementation for contact form
        if (window.notifications) {
            window.notifications.info('Contact Support', 'Email: support@resumeai.com<br>Phone: +1-234-567-8900<br>Hours: Mon-Fri 9AM-5PM', {
                persistent: true
            });
        }
    }
    
    showBugReportForm() {
        // Implementation for bug report form
        if (window.notifications) {
            window.notifications.info('Report a Bug', 'Please email bug reports to: bugs@resumeai.com<br>Include steps to reproduce the issue.', {
                persistent: true
            });
        }
    }
    
    navigateToSection(section) {
        const sectionElement = document.querySelector(`[data-section="${section}"]`);
        if (sectionElement) {
            this.showHelpPanel();
            this.toggleSection(sectionElement);
        }
    }
}

// Initialize help system
let helpSystem;

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        helpSystem = new HelpSystem();
        window.helpSystem = helpSystem;
    });
} else {
    helpSystem = new HelpSystem();
    window.helpSystem = helpSystem;
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HelpSystem;
}