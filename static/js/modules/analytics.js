// Analytics Module
const AnalyticsModule = {
    dateRange: null,
    candidateFlowChart: null,
    jobCategoriesChart: null,

    // Initialize analytics functionality
    init() {
        this.setupElements();
        this.setupEventListeners();
    },

    // Setup DOM elements
    setupElements() {
        this.dateRange = document.getElementById('dateRange');
    },

    // Setup event listeners
    setupEventListeners() {
        if (this.dateRange) {
            this.dateRange.addEventListener('change', () => {
                this.loadAnalytics();
            });
        }
    },

    // Load analytics data
    async loadAnalytics() {
        try {
            const days = this.dateRange ? this.dateRange.value : 30;
            const data = await APIService.analytics.getData(days);
            
            if (data.success) {
                this.updateMetricsCards(data.summary);
                this.updateMetricsTable(data.daily_stats);
                this.updateCharts(data.daily_stats);
            } else {
                throw new Error(data.error || 'Failed to load analytics data');
            }
        } catch (error) {
            console.error('Error loading analytics:', error);
            ToastUtils.showError('Failed to load analytics data');
        }
    },

    // Update metrics cards
    updateMetricsCards(summary) {
        const elements = {
            totalCandidates: document.getElementById('totalCandidates'),
            conversionRate: document.getElementById('conversionRate'),
            avgTimeToHire: document.getElementById('avgTimeToHire'),
            qualityOfHire: document.getElementById('qualityOfHire')
        };

        if (elements.totalCandidates) {
            elements.totalCandidates.textContent = summary.total_resumes || 0;
        }
        
        if (elements.conversionRate) {
            const conversionRate = summary.processed_resumes > 0 
                ? Math.round((summary.shortlisted / summary.processed_resumes) * 100) 
                : 0;
            elements.conversionRate.textContent = FormatUtils.formatPercentage(conversionRate);
        }
        
        if (elements.avgTimeToHire) {
            elements.avgTimeToHire.textContent = Math.round(summary.avg_processing_time || 0) + ' days';
        }
        
        if (elements.qualityOfHire) {
            const qualityOfHire = summary.total_resumes > 0 
                ? Math.round((summary.shortlisted / summary.total_resumes) * 100) 
                : 0;
            elements.qualityOfHire.textContent = FormatUtils.formatPercentage(qualityOfHire);
        }
    },

    // Update metrics table
    updateMetricsTable(dailyStats) {
        const tbody = document.getElementById('metricsTableBody');
        if (!tbody || !dailyStats || dailyStats.length === 0) return;

        const currentStats = dailyStats[dailyStats.length - 1];
        const previousStats = dailyStats[dailyStats.length - 2] || {
            total_resumes: 0,
            processed_resumes: 0,
            shortlisted: 0,
            rejected: 0
        };

        const metrics = [
            {
                name: 'Total Resumes',
                current: currentStats.total_resumes,
                previous: previousStats.total_resumes
            },
            {
                name: 'Processed Resumes',
                current: currentStats.processed_resumes,
                previous: previousStats.processed_resumes
            },
            {
                name: 'Shortlisted',
                current: currentStats.shortlisted,
                previous: previousStats.shortlisted
            },
            {
                name: 'Rejected',
                current: currentStats.rejected,
                previous: previousStats.rejected
            }
        ];

        tbody.innerHTML = metrics.map(metric => {
            const change = metric.current - metric.previous;
            const changePercent = metric.previous ? (change / metric.previous) * 100 : 0;
            const changeClass = change >= 0 ? 'text-success' : 'text-danger';
            const changeIcon = change >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';

            return `
                <tr>
                    <td>${metric.name}</td>
                    <td>${metric.current}</td>
                    <td>${metric.previous}</td>
                    <td class="${changeClass}">
                        <i class="fas ${changeIcon}"></i>
                        ${Math.abs(changePercent).toFixed(1)}%
                    </td>
                </tr>
            `;
        }).join('');
    },

    // Update charts
    updateCharts(dailyStats) {
        if (!dailyStats || dailyStats.length === 0) return;

        this.updateCandidateFlowChart(dailyStats);
        this.updateJobCategoriesChart(dailyStats);
    },

    // Update candidate flow chart
    updateCandidateFlowChart(dailyStats) {
        const candidateFlow = document.getElementById('candidateFlowChart');
        if (!candidateFlow) return;

        // Destroy existing chart if it exists
        if (this.candidateFlowChart) {
            this.candidateFlowChart.destroy();
        }
        
        const dates = dailyStats.map(stat => FormatUtils.formatDate(stat.date));
        const totalResumes = dailyStats.map(stat => stat.total_resumes);
        const processedResumes = dailyStats.map(stat => stat.processed_resumes);
        const shortlisted = dailyStats.map(stat => stat.shortlisted);

        this.candidateFlowChart = new Chart(candidateFlow, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [
                    {
                        label: 'Total Resumes',
                        data: totalResumes,
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37, 99, 235, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Processed',
                        data: processedResumes,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Shortlisted',
                        data: shortlisted,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Count'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    },

    // Update job categories chart
    updateJobCategoriesChart(dailyStats) {
        const jobCategories = document.getElementById('jobCategoriesChart');
        if (!jobCategories || dailyStats.length === 0) return;

        // Destroy existing chart if it exists
        if (this.jobCategoriesChart) {
            this.jobCategoriesChart.destroy();
        }
        
        const latestStats = dailyStats[dailyStats.length - 1];
        let categoryStats = {};
        
        try {
            categoryStats = JSON.parse(latestStats.job_category_stats || '{}');
        } catch (error) {
            console.error('Error parsing job category stats:', error);
            categoryStats = {};
        }
        
        const categories = Object.keys(categoryStats);
        const counts = Object.values(categoryStats);

        if (categories.length === 0) {
            jobCategories.innerHTML = '<p class="text-center text-muted">No job category data available</p>';
            return;
        }

        this.jobCategoriesChart = new Chart(jobCategories, {
            type: 'doughnut',
            data: {
                labels: categories,
                datasets: [{
                    data: counts,
                    backgroundColor: [
                        '#2563eb',
                        '#3b82f6',
                        '#60a5fa',
                        '#93c5fd',
                        '#bfdbfe',
                        '#1e40af',
                        '#1d4ed8'
                    ],
                    borderColor: '#ffffff',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = Math.round((value / total) * 100);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    },

    // Destroy charts on cleanup
    destroy() {
        if (this.candidateFlowChart) {
            this.candidateFlowChart.destroy();
            this.candidateFlowChart = null;
        }
        
        if (this.jobCategoriesChart) {
            this.jobCategoriesChart.destroy();
            this.jobCategoriesChart = null;
        }
    },

    // Export chart data
    exportData(format = 'csv') {
        // Implementation for exporting analytics data
        console.log(`Exporting analytics data in ${format} format`);
        ToastUtils.showInfo('Export functionality coming soon');
    },

    // Generate report
    generateReport() {
        // Implementation for generating analytics report
        console.log('Generating analytics report');
        ToastUtils.showInfo('Report generation coming soon');
    }
};

// Make globally available
window.AnalyticsModule = AnalyticsModule;

// Backward compatibility
window.loadAnalytics = AnalyticsModule.loadAnalytics.bind(AnalyticsModule);
window.updateMetricsCards = AnalyticsModule.updateMetricsCards.bind(AnalyticsModule);
window.updateMetricsTable = AnalyticsModule.updateMetricsTable.bind(AnalyticsModule);
window.updateCharts = AnalyticsModule.updateCharts.bind(AnalyticsModule);
