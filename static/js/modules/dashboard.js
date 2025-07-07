// Dashboard Module
const DashboardModule = {
    // Initialize dashboard functionality
    init() {
        this.loadDashboardData();
        this.setupRefreshInterval();
    },

    // Load all dashboard data
    async loadDashboardData() {
        try {
            await Promise.all([
                this.loadAnalyticsSummary(),
                this.loadRecentActivity(),
                this.loadTopCandidates(),
                this.loadJobCategoriesOverview()
            ]);
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            ToastUtils.showError('Failed to load dashboard data');
        }
    },

    // Load analytics summary
    async loadAnalyticsSummary() {
        try {
            const data = await APIService.analytics.getData(30);
            
            if (data.success) {
                this.updateDashboardStats(data.summary);
            }
        } catch (error) {
            console.error('Error loading analytics summary:', error);
        }
    },

    // Update dashboard statistics
    updateDashboardStats(summary) {
        const elements = {
            totalResumes: document.getElementById('totalResumes'),
            screenedResumes: document.getElementById('screenedResumes'),
            shortlisted: document.getElementById('shortlisted'),
            avgScreeningTime: document.getElementById('avgScreeningTime')
        };

        if (elements.totalResumes) {
            elements.totalResumes.textContent = summary.total_resumes || 0;
        }
        
        if (elements.screenedResumes) {
            elements.screenedResumes.textContent = summary.processed_resumes || 0;
        }
        
        if (elements.shortlisted) {
            elements.shortlisted.textContent = summary.shortlisted || 0;
        }
        
        if (elements.avgScreeningTime) {
            elements.avgScreeningTime.textContent = Math.round(summary.avg_processing_time || 0) + 'm';
        }
    },

    // Load recent activity
    async loadRecentActivity() {
        const activityList = document.getElementById('activityList');
        if (!activityList) return;

        try {
            const data = await APIService.candidates.getAll();
            
            if (data.success && data.candidates_by_job) {
                // Flatten candidates from all jobs and get recent ones
                const allCandidates = [];
                Object.values(data.candidates_by_job).forEach(jobData => {
                    allCandidates.push(...jobData.candidates);
                });
                
                // Sort by updated_at and take top 5
                allCandidates.sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));
                const recentCandidates = allCandidates.slice(0, 5);
                
                this.renderRecentActivity(recentCandidates, activityList);
            } else {
                activityList.innerHTML = '<p class="text-muted">No recent activity</p>';
            }
        } catch (error) {
            console.error('Error loading recent activity:', error);
            activityList.innerHTML = '<p class="text-muted">Failed to load recent activity</p>';
        }
    },

    // Render recent activity
    renderRecentActivity(candidates, container) {
        if (candidates.length === 0) {
            container.innerHTML = '<p class="text-muted">No recent activity</p>';
            return;
        }

        container.innerHTML = candidates.map(candidate => `
            <div class="activity-item">
                <div class="activity-icon ${candidate.status}">
                    <i class="fas ${this.getActivityIcon(candidate.status)}"></i>
                </div>
                <div class="activity-content">
                    <p class="activity-text">
                        <strong>${DOMUtils.escapeHtml(candidate.name || 'Anonymous')}</strong> - ${candidate.status}
                    </p>
                    <span class="activity-time">
                        ${FormatUtils.formatDate(candidate.updated_at)}
                    </span>
                </div>
            </div>
        `).join('');
    },

    // Get activity icon based on status
    getActivityIcon(status) {
        switch (status) {
            case 'shortlisted': return 'fa-check';
            case 'rejected': return 'fa-times';
            case 'pending': return 'fa-clock';
            default: return 'fa-user';
        }
    },

    // Load top candidates
    async loadTopCandidates() {
        const topCandidatesEl = document.getElementById('topCandidates');
        if (!topCandidatesEl) return;

        try {
            const data = await APIService.candidates.getAll();
            
            if (data.success && data.candidates_by_job) {
                // Flatten candidates and get top scoring ones
                const allCandidates = [];
                Object.values(data.candidates_by_job).forEach(jobData => {
                    allCandidates.push(...jobData.candidates);
                });
                
                // Sort by score and take top 5
                allCandidates.sort((a, b) => b.score - a.score);
                const topCandidates = allCandidates.slice(0, 5);
                
                this.renderTopCandidates(topCandidates, topCandidatesEl);
            } else {
                topCandidatesEl.innerHTML = '<p class="text-muted">No candidates yet</p>';
            }
        } catch (error) {
            console.error('Error loading top candidates:', error);
            topCandidatesEl.innerHTML = '<p class="text-muted">Failed to load top candidates</p>';
        }
    },

    // Render top candidates
    renderTopCandidates(candidates, container) {
        if (candidates.length === 0) {
            container.innerHTML = '<p class="text-muted">No candidates yet</p>';
            return;
        }

        container.innerHTML = candidates.map(candidate => `
            <div class="candidate-item" onclick="CandidatesModule.showCandidateDetails('${candidate.id}')">
                <div class="candidate-info">
                    <h4>${DOMUtils.escapeHtml(candidate.name || 'Anonymous')}</h4>
                    <p>${DOMUtils.escapeHtml(candidate.predicted_category || 'General')}</p>
                </div>
                <div class="candidate-score">
                    <span class="score">${Math.round(candidate.score || 0)}%</span>
                    <span class="match">Match</span>
                </div>
            </div>
        `).join('');
    },

    // Load job categories overview
    async loadJobCategoriesOverview() {
        const jobCategoriesGrid = document.getElementById('jobCategoriesGrid');
        if (!jobCategoriesGrid) return;

        try {
            const data = await APIService.jobs.getAll();
            
            if (data.success && data.jobs) {
                // Group jobs by category
                const categories = this.groupJobsByCategory(data.jobs);
                this.renderJobCategories(categories, jobCategoriesGrid);
            } else {
                jobCategoriesGrid.innerHTML = '<p class="text-muted">No job categories yet</p>';
            }
        } catch (error) {
            console.error('Error loading job categories:', error);
            jobCategoriesGrid.innerHTML = '<p class="text-muted">Failed to load job categories</p>';
        }
    },

    // Group jobs by category
    groupJobsByCategory(jobs) {
        const categories = {};
        
        jobs.forEach(job => {
            if (!categories[job.category]) {
                categories[job.category] = {
                    count: 0,
                    active: 0
                };
            }
            categories[job.category].count++;
            if (job.status === 'active' || !job.status) { // Assume active if no status
                categories[job.category].active++;
            }
        });
        
        return categories;
    },

    // Render job categories
    renderJobCategories(categories, container) {
        if (Object.keys(categories).length === 0) {
            container.innerHTML = '<p class="text-muted">No job categories yet</p>';
            return;
        }

        container.innerHTML = Object.entries(categories).map(([category, stats]) => `
            <div class="category-card" onclick="NavigationModule.showSection('jobs')">
                <div class="category-icon">
                    <i class="fas fa-briefcase"></i>
                </div>
                <div class="category-info">
                    <h4>${DOMUtils.escapeHtml(category)}</h4>
                    <p>${stats.count} job${stats.count !== 1 ? 's' : ''}</p>
                    <span class="active-count">${stats.active} active</span>
                </div>
            </div>
        `).join('');
    },

    // Setup auto-refresh interval
    setupRefreshInterval() {
        // Refresh dashboard data every 5 minutes
        setInterval(() => {
            const currentSection = window.location.hash.slice(1) || 'dashboard';
            if (currentSection === 'dashboard') {
                this.loadDashboardData();
            }
        }, 5 * 60 * 1000); // 5 minutes
    },

    // Refresh specific section
    async refreshSection(section) {
        switch (section) {
            case 'analytics':
                await this.loadAnalyticsSummary();
                break;
            case 'activity':
                await this.loadRecentActivity();
                break;
            case 'candidates':
                await this.loadTopCandidates();
                break;
            case 'jobs':
                await this.loadJobCategoriesOverview();
                break;
            default:
                await this.loadDashboardData();
        }
    },

    // Get dashboard metrics
    async getDashboardMetrics() {
        try {
            const [analyticsData, candidatesData, jobsData] = await Promise.all([
                APIService.analytics.getData(30),
                APIService.candidates.getAll(),
                APIService.jobs.getAll()
            ]);

            return {
                analytics: analyticsData.success ? analyticsData.summary : null,
                candidates: candidatesData.success ? candidatesData : null,
                jobs: jobsData.success ? jobsData.jobs : null
            };
        } catch (error) {
            console.error('Error getting dashboard metrics:', error);
            return null;
        }
    }
};

// Make globally available
window.DashboardModule = DashboardModule;

// Backward compatibility
window.loadDashboardData = DashboardModule.loadDashboardData.bind(DashboardModule);
window.refreshAllData = DashboardModule.loadDashboardData.bind(DashboardModule);
