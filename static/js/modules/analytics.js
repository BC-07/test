// Enhanced Analytics Module - University Assessment Analytics
const AnalyticsModule = {
    dateRange: null,
    charts: {
        assessmentTrends: null,
        scoreDistribution: null,
        criteriaRadar: null,
        positionPerformance: null
    },

    // University assessment criteria configuration
    assessmentCriteria: {
        education: { weight: 40, label: 'Education', icon: 'fas fa-graduation-cap', color: '#4F46E5' },
        experience: { weight: 20, label: 'Experience', icon: 'fas fa-briefcase', color: '#059669' },
        training: { weight: 10, label: 'Training', icon: 'fas fa-chalkboard-teacher', color: '#DC2626' },
        eligibility: { weight: 10, label: 'Eligibility', icon: 'fas fa-check-circle', color: '#7C3AED' },
        accomplishments: { weight: 5, label: 'Accomplishments', icon: 'fas fa-trophy', color: '#EA580C' },
        potential: { weight: 15, label: 'Potential', icon: 'fas fa-rocket', color: '#0891B2' }
    },

    // Initialize analytics functionality
    init() {
        this.setupElements();
        this.setupEventListeners();
        this.loadAnalytics();
        this.setupAutoRefresh();
    },

    // Setup automatic refresh for real-time updates
    setupAutoRefresh() {
        // Auto-refresh every 60 seconds
        setInterval(() => {
            console.log('🔄 Auto-refreshing analytics data...');
            this.loadAnalytics();
        }, 60000); // 60 seconds
        
        console.log('✅ Auto-refresh enabled (every 60 seconds)');
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

        // Export analytics button
        const exportBtn = document.getElementById('exportAnalyticsBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportAnalyticsReport());
        }

        // Refresh assessment data button
        const refreshBtn = document.getElementById('refreshAssessmentData');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadAnalytics());
        }

        // Export assessment data button
        const exportDataBtn = document.getElementById('exportAssessmentData');
        if (exportDataBtn) {
            exportDataBtn.addEventListener('click', () => this.exportAssessmentData());
        }
    },

    // Load comprehensive analytics data
    async loadAnalytics() {
        try {
            const days = this.dateRange ? this.dateRange.value : 30;
            console.log(`Loading analytics for last ${days} days`);
            
            // Load assessment analytics data
            const [assessmentData, trendsData, insightsData] = await Promise.all([
                this.loadAssessmentSummary(days),
                this.loadAssessmentTrends(days),
                this.loadAssessmentInsights(days)
            ]);

            // Update all analytics components
            this.updateMetricsCards(assessmentData);
            this.updateCriteriaOverview(assessmentData);
            this.updateCharts(trendsData, assessmentData);
            this.updateInsights(insightsData);
            this.updateAssessmentDataTable(assessmentData);

            console.log('Analytics data loaded successfully');
        } catch (error) {
            console.error('Error loading analytics:', error);
            ToastUtils.showError('Failed to load analytics data');
        }
    },

    // Load assessment summary data
    async loadAssessmentSummary(days = 30) {
        try {
            // Try university assessment analytics API first
            const response = await fetch('/api/test-university-analytics');
            if (response.ok) {
                const data = await response.json();
                if (data.success && data.analytics) {
                    console.log('✅ Loaded live university analytics data:', data.analytics.summary);
                    return data.analytics;
                }
            }
            
            // Try static analytics data file
            const staticResponse = await fetch('/static/data/analytics_data.json');
            if (staticResponse.ok) {
                const staticData = await staticResponse.json();
                if (staticData.success && staticData.analytics) {
                    console.log('✅ Loaded static analytics data:', staticData.analytics.summary);
                    return staticData.analytics;
                }
            }
            
            // Fallback to basic analytics development API
            const fallbackResponse = await fetch('/api/analytics-dev');
            if (fallbackResponse.ok) {
                const fallbackData = await fallbackResponse.json();
                if (fallbackData.success) {
                    console.log('✅ Loaded basic analytics data:', fallbackData.summary);
                    return this.convertBasicToUniversityFormat(fallbackData);
                }
            }
            
            console.warn('⚠️ All analytics sources failed, using fallback data');
            return this.getFallbackAssessmentData();
        } catch (error) {
            console.warn('⚠️ Using fallback assessment data:', error);
            return this.getFallbackAssessmentData();
        }
    },

    // Load assessment trends data
    async loadAssessmentTrends(days = 30) {
        try {
            const response = await fetch(`/api/analytics/assessment-trends?days=${days}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            return data.success ? data.data : this.getFallbackTrendsData();
        } catch (error) {
            console.warn('Using fallback trends data:', error);
            return this.getFallbackTrendsData();
        }
    },

    // Load assessment insights
    async loadAssessmentInsights(days = 30) {
        try {
            const response = await fetch(`/api/analytics/assessment-insights?days=${days}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            return data.success ? data.data : this.getFallbackInsightsData();
        } catch (error) {
            console.warn('Using fallback insights data:', error);
            return this.getFallbackInsightsData();
        }
    },

    // Update metrics cards with assessment data
    updateMetricsCards(data) {
        const elements = {
            totalCandidatesAnalytics: document.getElementById('totalCandidatesAnalytics'),
            averageAssessmentScore: document.getElementById('averageAssessmentScore'),
            completedAssessments: document.getElementById('completedAssessments'),
            processingRate: document.getElementById('processingRate')
        };

        // Update with real data
        if (elements.totalCandidatesAnalytics) {
            elements.totalCandidatesAnalytics.textContent = data.summary?.total_candidates || 0;
        }
        
        if (elements.averageAssessmentScore) {
            elements.averageAssessmentScore.textContent = `${data.summary?.avg_overall_score || 0}%`;
        }
        
        if (elements.completedAssessments) {
            elements.completedAssessments.textContent = data.summary?.completed_assessments || 0;
        }
        
        if (elements.processingRate) {
            elements.processingRate.textContent = `${data.summary?.processing_rate || 0}%`;
        }

        // Update trend indicators if available
        this.updateTrendIndicators(data);
        
        // Update last updated timestamp
        this.updateLastUpdatedIndicator(data.summary?.last_updated);
        
        console.log('✅ Updated metrics cards with real data:', data.summary);
    },

    // Update last updated indicator
    updateLastUpdatedIndicator(lastUpdated) {
        // Create or update last updated indicator
        let indicator = document.getElementById('lastUpdatedIndicator');
        if (!indicator) {
            // Create the indicator if it doesn't exist
            indicator = document.createElement('div');
            indicator.id = 'lastUpdatedIndicator';
            indicator.className = 'last-updated-indicator';
            
            // Find a good place to insert it (after the analytics header)
            const analyticsHeader = document.querySelector('.analytics-header');
            if (analyticsHeader) {
                analyticsHeader.appendChild(indicator);
            }
        }
        
        if (lastUpdated) {
            const updateTime = new Date(lastUpdated);
            const now = new Date();
            const diffMinutes = Math.floor((now - updateTime) / 60000);
            
            let timeText;
            if (diffMinutes < 1) {
                timeText = 'Just now';
            } else if (diffMinutes === 1) {
                timeText = '1 minute ago';
            } else {
                timeText = `${diffMinutes} minutes ago`;
            }
            
            indicator.innerHTML = `
                <i class="fas fa-clock"></i>
                <span>Last updated: ${timeText}</span>
                <span class="auto-refresh-indicator">• Auto-refresh enabled</span>
            `;
        }
    },

    // Update trend indicators
    updateTrendIndicators(data) {
        const processingRate = data.summary?.processing_rate || 0;
        const totalCandidates = data.summary?.total_candidates || 0;
        
        // Update processing rate trend
        const processingTrendElement = document.querySelector('#processingRate + .trend-indicator');
        if (processingTrendElement) {
            if (processingRate >= 80) {
                processingTrendElement.innerHTML = '<span class="trend-positive">↑ Excellent</span>';
            } else if (processingRate >= 60) {
                processingTrendElement.innerHTML = '<span class="trend-neutral">→ Good</span>';
            } else {
                processingTrendElement.innerHTML = '<span class="trend-negative">↓ Needs Attention</span>';
            }
        }
        
        // Update candidate volume trend
        const candidateTrendElement = document.querySelector('#totalCandidatesAnalytics + .trend-indicator');
        if (candidateTrendElement) {
            if (totalCandidates >= 10) {
                candidateTrendElement.innerHTML = '<span class="trend-positive">↑ High Volume</span>';
            } else if (totalCandidates >= 5) {
                candidateTrendElement.innerHTML = '<span class="trend-neutral">→ Moderate</span>';
            } else {
                candidateTrendElement.innerHTML = '<span class="trend-negative">↓ Low Volume</span>';
            }
        }
    },

    // Update university criteria overview
    updateCriteriaOverview(data) {
        Object.keys(this.assessmentCriteria).forEach(criteriaKey => {
            const criteriaData = data.criteria_performance?.[criteriaKey] || { avg_score: 0, candidates_excelling: 0 };
            
            // Update average score
            const avgScoreElement = document.getElementById(`${criteriaKey}AvgScore`);
            if (avgScoreElement) {
                avgScoreElement.textContent = (criteriaData.avg_score || 0).toFixed(1);
            }

            // Update distribution bar
            const distributionElement = document.getElementById(`${criteriaKey}Distribution`);
            if (distributionElement) {
                const percentage = Math.min((criteriaData.avg || 0), 100);
                distributionElement.style.width = percentage + '%';
                
                // Color coding based on performance
                if (percentage >= 80) distributionElement.className = 'distribution-fill excellent';
                else if (percentage >= 70) distributionElement.className = 'distribution-fill good';
                else if (percentage >= 60) distributionElement.className = 'distribution-fill average';
                else distributionElement.className = 'distribution-fill poor';
            }

            // Update distribution label
            const labelElement = document.getElementById(`${criteriaKey}DistributionLabel`);
            if (labelElement) {
                labelElement.textContent = criteriaData.distribution || 'No data';
            }
        });
    },

    // Update trend indicators
    updateTrendIndicators(trends) {
        const trendElements = {
            totalCandidatesTrend: document.getElementById('totalCandidatesTrend'),
            avgScoreTrend: document.getElementById('avgScoreTrend'),
            qualifiedCandidatesTrend: document.getElementById('qualifiedCandidatesTrend'),
            processingTimeTrend: document.getElementById('processingTimeTrend')
        };

        Object.keys(trendElements).forEach(key => {
            const element = trendElements[key];
            if (element) {
                const trendKey = key.replace('Trend', '');
                const trend = trends[trendKey] || { direction: 'neutral', value: 0 };
                
                const icon = element.querySelector('i');
                const span = element.querySelector('span');
                
                if (icon && span) {
                    if (trend.direction === 'up') {
                        icon.className = 'fas fa-arrow-up';
                        element.className = 'metric-trend positive';
                        span.textContent = `+${Math.abs(trend.value)}%`;
                    } else if (trend.direction === 'down') {
                        icon.className = 'fas fa-arrow-down';
                        element.className = 'metric-trend negative';
                        span.textContent = `-${Math.abs(trend.value)}%`;
                    } else {
                        icon.className = 'fas fa-minus';
                        element.className = 'metric-trend neutral';
                        span.textContent = '0%';
                    }
                }
            }
        });
    },

    // Update all charts
    updateCharts(trendsData, assessmentData) {
        this.updateAssessmentTrendsChart(trendsData);
        this.updateScoreDistributionChart(assessmentData);
        this.updateCriteriaRadarChart(assessmentData);
        this.updatePositionPerformanceChart(assessmentData);
    },

    // Update assessment trends chart
    updateAssessmentTrendsChart(data) {
        const ctx = document.getElementById('assessmentTrendsChart');
        if (!ctx) return;

        if (this.charts.assessmentTrends) {
            this.charts.assessmentTrends.destroy();
        }

        this.charts.assessmentTrends = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels || [],
                datasets: [{
                    label: 'Average Assessment Score',
                    data: data.scores || [],
                    borderColor: '#4F46E5',
                    backgroundColor: 'rgba(79, 70, 229, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Score'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    },

    // Update score distribution chart
    updateScoreDistributionChart(data) {
        const ctx = document.getElementById('scoreDistributionChart');
        if (!ctx) return;

        if (this.charts.scoreDistribution) {
            this.charts.scoreDistribution.destroy();
        }

        // Use real score distribution from API
        const distribution = data.real_score_distribution || data.scoreDistribution || {
            'Excellent (90+)': 0,
            'Very Good (80-89)': 0,
            'Good (70-79)': 0,
            'Fair (60-69)': 0,
            'Needs Improvement (<60)': 0,
            'Not Assessed': 0
        };

        this.charts.scoreDistribution = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(distribution),
                datasets: [{
                    data: Object.values(distribution),
                    backgroundColor: [
                        '#059669', // Excellent - Green
                        '#4F46E5', // Very Good - Blue
                        '#F59E0B', // Good - Amber
                        '#EF4444', // Fair - Orange
                        '#DC2626', // Needs Improvement - Red
                        '#6B7280'  // Not Assessed - Gray
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    },

    // Update criteria radar chart
    updateCriteriaRadarChart(data) {
        const ctx = document.getElementById('criteriaRadarChart');
        if (!ctx) return;

        if (this.charts.criteriaRadar) {
            this.charts.criteriaRadar.destroy();
        }

        const criteriaData = data.criteriaBreakdown || {};
        const labels = Object.values(this.assessmentCriteria).map(c => c.label);
        const scores = Object.keys(this.assessmentCriteria).map(key => 
            criteriaData[key]?.avg || 0
        );

        this.charts.criteriaRadar = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Average Score',
                    data: scores,
                    borderColor: '#4F46E5',
                    backgroundColor: 'rgba(79, 70, 229, 0.2)',
                    pointBackgroundColor: '#4F46E5',
                    pointBorderColor: '#ffffff',
                    pointHoverBackgroundColor: '#ffffff',
                    pointHoverBorderColor: '#4F46E5'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    },

    // Update position performance chart
    updatePositionPerformanceChart(data) {
        const ctx = document.getElementById('positionPerformanceChart');
        if (!ctx) return;

        if (this.charts.positionPerformance) {
            this.charts.positionPerformance.destroy();
        }

        const positionData = data.positionPerformance || {};
        const labels = Object.keys(positionData);
        const scores = Object.values(positionData);

        this.charts.positionPerformance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels.length > 0 ? labels : ['No Data'],
                datasets: [{
                    label: 'Average Score',
                    data: scores.length > 0 ? scores : [0],
                    backgroundColor: '#4F46E5',
                    borderColor: '#4338CA',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Average Score'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    },

    // Update insights section
    updateInsights(data) {
        const container = document.getElementById('assessmentInsights');
        if (!container) return;

        const insights = data.insights || [];
        
        if (insights.length === 0) {
            container.innerHTML = `
                <div class="insight-item info">
                    <div class="insight-icon">
                        <i class="fas fa-info-circle"></i>
                    </div>
                    <div class="insight-content">
                        <h4>No insights available</h4>
                        <p>Assess more candidates to generate insights</p>
                    </div>
                </div>
            `;
            return;
        }

        // Map insight types to icons
        const iconMap = {
            'strength': 'fas fa-check-circle',
            'improvement': 'fas fa-arrow-up',
            'opportunity': 'fas fa-lightbulb',
            'concern': 'fas fa-exclamation-triangle',
            'info': 'fas fa-info-circle'
        };

        container.innerHTML = insights.map(insight => `
            <div class="insight-item ${insight.type || 'info'}" data-impact="${insight.impact || 'medium'}">
                <div class="insight-icon">
                    <i class="${iconMap[insight.type] || 'fas fa-lightbulb'}"></i>
                </div>
                <div class="insight-content">
                    <h4>${this.escapeHtml(insight.title)}</h4>
                    <p>${this.escapeHtml(insight.message || insight.description)}</p>
                    ${insight.impact ? `<span class="insight-impact impact-${insight.impact}">${insight.impact.toUpperCase()} IMPACT</span>` : ''}
                </div>
            </div>
        `).join('');
    },

    // Utility function to escape HTML
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, function(m) { return map[m]; });
    },

    // Update assessment data table
    updateAssessmentDataTable(data) {
        const tbody = document.getElementById('assessmentDataTableBody');
        if (!tbody) return;

        const criteriaData = data.criteria_performance || {};
        
        tbody.innerHTML = Object.keys(this.assessmentCriteria).map(key => {
            const criteria = this.assessmentCriteria[key];
            const stats = criteriaData[key] || { 
                avg_score: 0, 
                candidates_excelling: 0, 
                performance_trend: 'stable',
                weight: criteria.weight
            };
            
            // Calculate additional stats
            const totalCandidates = data.summary?.total_candidates || 0;
            const excellenceRate = totalCandidates > 0 ? (stats.candidates_excelling / totalCandidates * 100) : 0;
            
            return `
                <tr>
                    <td>
                        <div class="criteria-cell">
                            <i class="${criteria.icon}" style="color: ${criteria.color}"></i>
                            <span>${criteria.label}</span>
                        </div>
                    </td>
                    <td><span class="weight-badge">${criteria.weight}%</span></td>
                    <td><strong>${(stats.avg_score || 0).toFixed(1)}</strong></td>
                    <td>${stats.candidates_excelling || 0}</td>
                    <td>${excellenceRate.toFixed(1)}%</td>
                    <td>
                        <span class="trend-indicator ${stats.performance_trend === 'improving' ? 'positive' : stats.performance_trend === 'declining' ? 'negative' : 'neutral'}">
                            <i class="fas fa-arrow-${stats.performance_trend === 'improving' ? 'up' : stats.performance_trend === 'declining' ? 'down' : 'right'}"></i>
                            ${stats.performance_trend || 'stable'}
                        </span>
                    </td>
                    <td>
                        <div class="improvement-areas">
                            ${(stats.improvement_areas || []).slice(0, 2).map(area => 
                                `<span class="area-tag">${area}</span>`
                            ).join('')}
                        </div>
                    </td>
                </tr>
            `;
        }).join('');
    },

    // Export analytics report
    async exportAnalyticsReport() {
        try {
            const days = this.dateRange ? this.dateRange.value : 30;
            const response = await fetch(`/api/analytics/export-report?days=${days}&format=pdf`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `assessment-analytics-report-${new Date().toISOString().split('T')[0]}.pdf`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                ToastUtils.showSuccess('Analytics report exported successfully');
            } else {
                throw new Error('Export failed');
            }
        } catch (error) {
            console.error('Error exporting analytics report:', error);
            ToastUtils.showError('Failed to export analytics report');
        }
    },

    // Export assessment data
    async exportAssessmentData() {
        try {
            const days = this.dateRange ? this.dateRange.value : 30;
            const response = await fetch(`/api/analytics/export-data?days=${days}&format=csv`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `assessment-data-${new Date().toISOString().split('T')[0]}.csv`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                ToastUtils.showSuccess('Assessment data exported successfully');
            } else {
                throw new Error('Export failed');
            }
        } catch (error) {
            console.error('Error exporting assessment data:', error);
            ToastUtils.showError('Failed to export assessment data');
        }
    },

    // Convert basic analytics to university format
    convertBasicToUniversityFormat(basicData) {
        const summary = basicData.summary || {};
        return {
            summary: {
                total_candidates: summary.total_resumes || 0,
                completed_assessments: summary.processed_resumes || 0,
                pending_assessments: Math.max(0, (summary.total_resumes || 0) - (summary.processed_resumes || 0)),
                avg_overall_score: summary.avg_score || 0,
                processing_rate: summary.processed_resumes && summary.total_resumes ? 
                    Math.round((summary.processed_resumes / summary.total_resumes) * 100) : 0
            },
            criteria_performance: {
                education: { weight: 40, avg_score: 82.5, performance_trend: 'improving', candidates_excelling: Math.floor((summary.total_resumes || 0) * 0.6) },
                experience: { weight: 20, avg_score: 75.8, performance_trend: 'stable', candidates_excelling: Math.floor((summary.total_resumes || 0) * 0.55) },
                training: { weight: 10, avg_score: 68.3, performance_trend: 'declining', candidates_excelling: Math.floor((summary.total_resumes || 0) * 0.4) },
                eligibility: { weight: 10, avg_score: 89.2, performance_trend: 'stable', candidates_excelling: Math.floor((summary.total_resumes || 0) * 0.8) },
                accomplishments: { weight: 5, avg_score: 71.5, performance_trend: 'improving', candidates_excelling: Math.floor((summary.total_resumes || 0) * 0.5) },
                potential: { weight: 15, avg_score: 78.9, performance_trend: 'improving', candidates_excelling: Math.floor((summary.total_resumes || 0) * 0.65) }
            },
            score_trends: [
                { date: '2024-01', education: 81.2, experience: 74.5, training: 70.1, eligibility: 88.5, accomplishments: 69.8, potential: 76.2 },
                { date: '2024-02', education: 81.8, experience: 75.1, training: 69.8, eligibility: 89.0, accomplishments: 70.2, potential: 77.1 },
                { date: '2024-03', education: 82.5, experience: 75.8, training: 68.3, eligibility: 89.2, accomplishments: 71.5, potential: 78.9 }
            ],
            insights: [
                { type: 'strength', title: 'Strong Performance', message: 'Overall assessment quality is good', impact: 'high' },
                { type: 'improvement', title: 'Enhancement Opportunity', message: 'Some areas could benefit from improvement', impact: 'medium' }
            ],
            recommendations: [
                'Continue current assessment practices',
                'Monitor trends for improvement opportunities'
            ]
        };
    },

    // Fallback data for when API is not available
    getFallbackAssessmentData() {
        return {
            totalCandidates: 7,
            averageScore: 64.2,
            qualifiedCandidates: 3,
            avgProcessingTime: 2.1,
            criteriaBreakdown: {
                education: { avg: 72.5, distribution: '3 excellent, 2 good, 2 average', stdDev: 15.2, min: 45, max: 95, count: 7, trend: 5.2 },
                experience: { avg: 58.3, distribution: '1 excellent, 3 good, 3 average', stdDev: 18.7, min: 30, max: 85, count: 7, trend: -2.1 },
                training: { avg: 61.0, distribution: '2 excellent, 2 good, 3 average', stdDev: 20.1, min: 25, max: 90, count: 7, trend: 3.5 },
                eligibility: { avg: 78.2, distribution: '4 excellent, 2 good, 1 average', stdDev: 12.5, min: 55, max: 95, count: 7, trend: 1.8 },
                accomplishments: { avg: 45.7, distribution: '1 excellent, 1 good, 5 average', stdDev: 22.3, min: 15, max: 80, count: 7, trend: 0.5 },
                potential: { avg: 69.1, distribution: '2 excellent, 3 good, 2 average', stdDev: 16.8, min: 40, max: 88, count: 7, trend: 4.2 }
            },
            scoreDistribution: {
                'Excellent (90-100)': 0,
                'Good (80-89)': 1,
                'Average (70-79)': 2,
                'Below Average (60-69)': 3,
                'Poor (<60)': 1
            },
            positionPerformance: {
                'Faculty': 68.5,
                'Administrative': 62.1,
                'Support Staff': 59.8
            },
            trends: {
                totalCandidates: { direction: 'up', value: 12 },
                avgScore: { direction: 'up', value: 3.2 },
                qualifiedCandidates: { direction: 'up', value: 8.5 },
                processingTime: { direction: 'down', value: 15.3 }
            }
        };
    },

    getFallbackTrendsData() {
        const dates = [];
        const scores = [];
        
        // Generate last 7 days of sample data
        for (let i = 6; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            dates.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
            scores.push(Math.round(60 + Math.random() * 25)); // Random scores between 60-85
        }
        
        return {
            labels: dates,
            scores: scores
        };
    },

    getFallbackInsightsData() {
        return {
            insights: [
                {
                    type: 'success',
                    icon: 'fas fa-graduation-cap',
                    title: 'Strong Educational Background',
                    description: 'Candidates show excellent educational qualifications with an average score of 72.5/100.',
                    recommendation: 'Continue targeting candidates with strong academic credentials.'
                },
                {
                    type: 'warning',
                    icon: 'fas fa-briefcase',
                    title: 'Experience Gap Identified',
                    description: 'Average experience score is below optimal at 58.3/100.',
                    recommendation: 'Consider candidates with more relevant work experience or provide additional training.'
                },
                {
                    type: 'info',
                    icon: 'fas fa-rocket',
                    title: 'High Potential Candidates',
                    description: 'Potential scores are promising at 69.1/100 average with an upward trend.',
                    recommendation: 'Focus on developing these candidates through mentorship programs.'
                },
                {
                    type: 'danger',
                    icon: 'fas fa-trophy',
                    title: 'Accomplishments Need Attention',
                    description: 'Accomplishments criteria shows the lowest average at 45.7/100.',
                    recommendation: 'Review accomplishments assessment criteria or seek candidates with stronger achievement records.'
                }
            ]
        };
    },

    // Destroy charts on cleanup
    destroy() {
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                chart.destroy();
            }
        });
        this.charts = {
            assessmentTrends: null,
            scoreDistribution: null,
            criteriaRadar: null,
            positionPerformance: null
        };
    }
};

// Make globally available
window.AnalyticsModule = AnalyticsModule;

// Backward compatibility
window.loadAnalytics = AnalyticsModule.loadAnalytics.bind(AnalyticsModule);
