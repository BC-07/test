// Candidates Module
const CandidatesModule = {
    candidatesContent: null,
    modal: null,
    searchInput: null,
    sortSelect: null,
    filterSelect: null,
    candidatesData: null,
    selectedCandidates: new Set(),
    isLoading: false,

    // Initialize candidates functionality
    init() {
        this.setupElements();
        this.setupEventListeners();
        this.initializeFilters();
    },

    // Setup DOM elements
    setupElements() {
        this.candidatesContent = document.getElementById('candidatesContent');
        this.searchInput = document.getElementById('candidateSearch');
        this.sortSelect = document.getElementById('candidateSort');
        this.filterSelect = document.getElementById('candidateFilter');
        
        if (document.getElementById('candidateDetailsModal')) {
            this.modal = new bootstrap.Modal(document.getElementById('candidateDetailsModal'));
        }
    },

    // Setup event listeners
    setupEventListeners() {
        // Refresh button
        const refreshBtn = document.getElementById('refreshCandidates');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadCandidates();
            });
        }

        // Export button
        const exportBtn = document.getElementById('exportCandidates');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.exportCandidates();
            });
        }

        // Search functionality
        if (this.searchInput) {
            this.searchInput.addEventListener('input', this.debounce(() => {
                this.filterAndDisplayCandidates();
            }, 300));
        }

        // Sort functionality
        if (this.sortSelect) {
            this.sortSelect.addEventListener('change', () => {
                this.filterAndDisplayCandidates();
            });
        }

        // Filter functionality
        if (this.filterSelect) {
            this.filterSelect.addEventListener('change', () => {
                this.filterAndDisplayCandidates();
            });
        }

        // Clear filters
        const clearFiltersBtn = document.getElementById('clearFilters');
        if (clearFiltersBtn) {
            clearFiltersBtn.addEventListener('click', () => {
                this.clearFilters();
            });
        }

        // Bulk actions
        const bulkActionsBtn = document.getElementById('bulkActions');
        if (bulkActionsBtn) {
            bulkActionsBtn.addEventListener('click', () => {
                this.showBulkActionsMenu();
            });
        }

        // Modal action buttons
        this.setupModalActions();
    },

    // Initialize filters and controls
    initializeFilters() {
        // Set default values if elements exist
        if (this.sortSelect) {
            this.sortSelect.value = 'score-desc';
        }
        if (this.filterSelect) {
            this.filterSelect.value = 'all';
        }
    },

    // Debounce utility for search
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Load candidates from API
    async loadCandidates() {
        if (!this.candidatesContent) return;

        this.setLoadingState(true);

        try {
            const data = await APIService.candidates.getAll();
            
            if (data.success) {
                this.candidatesData = data.candidates_by_job;
                this.totalCandidates = data.total_candidates;
                this.filterAndDisplayCandidates();
                this.updateCandidateStats();
            } else {
                ToastUtils.showError('Failed to load candidates');
            }
        } catch (error) {
            console.error('Error loading candidates:', error);
            ToastUtils.showError('Error loading candidates');
        } finally {
            this.setLoadingState(false);
        }
    },

    // Filter and display candidates based on search, sort, and filter criteria
    filterAndDisplayCandidates() {
        if (!this.candidatesData) return;

        let filteredData = { ...this.candidatesData };
        const searchTerm = this.searchInput ? this.searchInput.value.toLowerCase().trim() : '';
        const sortBy = this.sortSelect ? this.sortSelect.value : 'score-desc';
        const statusFilter = this.filterSelect ? this.filterSelect.value : 'all';

        // Apply filters to each job category
        Object.keys(filteredData).forEach(jobId => {
            let candidates = filteredData[jobId].candidates;

            // Apply search filter
            if (searchTerm) {
                candidates = candidates.filter(candidate => 
                    candidate.name.toLowerCase().includes(searchTerm) ||
                    candidate.email.toLowerCase().includes(searchTerm) ||
                    candidate.predicted_category.toLowerCase().includes(searchTerm) ||
                    (candidate.all_skills || []).some(skill => 
                        skill.toLowerCase().includes(searchTerm)
                    )
                );
            }

            // Apply status filter
            if (statusFilter !== 'all') {
                candidates = candidates.filter(candidate => 
                    candidate.status.toLowerCase() === statusFilter
                );
            }

            // Apply sorting
            candidates = this.sortCandidates(candidates, sortBy);

            filteredData[jobId].candidates = candidates;
        });

        this.displayCandidatesByJob(filteredData, this.totalCandidates);
        this.setupCandidateActionListeners();
    },

    // Sort candidates based on criteria
    sortCandidates(candidates, sortBy) {
        return [...candidates].sort((a, b) => {
            switch (sortBy) {
                case 'name-asc':
                    return a.name.localeCompare(b.name);
                case 'name-desc':
                    return b.name.localeCompare(a.name);
                case 'score-asc':
                    return a.score - b.score;
                case 'score-desc':
                    return b.score - a.score;
                case 'category-asc':
                    return a.predicted_category.localeCompare(b.predicted_category);
                case 'category-desc':
                    return b.predicted_category.localeCompare(a.predicted_category);
                case 'status-asc':
                    return a.status.localeCompare(b.status);
                case 'status-desc':
                    return b.status.localeCompare(a.status);
                default:
                    return b.score - a.score; // Default to score desc
            }
        });
    },

    // Set loading state
    setLoadingState(isLoading) {
        this.isLoading = isLoading;
        const refreshBtn = document.getElementById('refreshCandidates');
        
        if (refreshBtn) {
            if (isLoading) {
                refreshBtn.disabled = true;
                refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Loading...';
            } else {
                refreshBtn.disabled = false;
                refreshBtn.innerHTML = '<i class="fas fa-sync me-2"></i>Refresh';
            }
        }

        if (isLoading && this.candidatesContent) {
            this.candidatesContent.innerHTML = `
                <div class="loading-state">
                    <div class="loading-spinner">
                        <i class="fas fa-spinner fa-spin"></i>
                    </div>
                    <p>Loading candidates...</p>
                </div>
            `;
        }
    },

    // Update candidate statistics
    updateCandidateStats() {
        const statsContainer = document.getElementById('candidateStats');
        if (!statsContainer || !this.candidatesData) return;

        const totalCandidates = Object.values(this.candidatesData)
            .reduce((sum, job) => sum + job.candidates.length, 0);
        
        const statusCounts = {};
        Object.values(this.candidatesData).forEach(job => {
            job.candidates.forEach(candidate => {
                statusCounts[candidate.status] = (statusCounts[candidate.status] || 0) + 1;
            });
        });

        statsContainer.innerHTML = `
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">${totalCandidates}</div>
                    <div class="stat-label">Total Candidates</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${statusCounts.pending || 0}</div>
                    <div class="stat-label">Pending</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${statusCounts.shortlisted || 0}</div>
                    <div class="stat-label">Shortlisted</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${statusCounts.rejected || 0}</div>
                    <div class="stat-label">Rejected</div>
                </div>
            </div>
        `;
    },

    // Display candidates grouped by job
    displayCandidatesByJob(candidatesByJob, totalCandidates) {
        this.candidatesContent.innerHTML = '';
        
        if (totalCandidates === 0) {
            this.candidatesContent.innerHTML = `
                <div class="no-candidates-message">
                    <div class="no-candidates-icon">
                        <i class="fas fa-users"></i>
                    </div>
                    <h4>No Candidates Found</h4>
                    <p>Upload some resumes in the "Upload Resume" section to see candidates here.</p>
                    <a href="#upload" class="btn btn-primary" onclick="NavigationModule.showSection('upload')">
                        <i class="fas fa-upload me-2"></i>Upload Resumes
                    </a>
                </div>
            `;
            return;
        }
        
        // Create content for each job category
        Object.entries(candidatesByJob).forEach(([jobId, jobData]) => {
            if (jobData.candidates.length === 0) return;
            
            const jobSection = this.createJobSection(jobData);
            this.candidatesContent.appendChild(jobSection);
        });
    },

    // Create job section element
    createJobSection(jobData) {
        const jobSection = document.createElement('div');
        jobSection.className = 'job-section';
        
        // Handle LSPU job structure vs legacy/unassigned
        const isLSPUJob = jobData.position_title && jobData.campus_name;
        const jobTitle = isLSPUJob ? jobData.position_title : (jobData.job_title || 'Unassigned Candidates');
        const jobCategory = isLSPUJob ? jobData.position_category : (jobData.job_category || 'General');
        
        jobSection.innerHTML = `
            <div class="job-header">
                <h3 class="job-title">
                    <i class="fas fa-briefcase me-2"></i>
                    ${DOMUtils.escapeHtml(jobTitle)}
                </h3>
                <div class="job-meta">
                    <span class="badge bg-primary">${DOMUtils.escapeHtml(jobCategory)}</span>
                    ${isLSPUJob ? `<span class="badge bg-info">${DOMUtils.escapeHtml(jobData.campus_name)}</span>` : ''}
                    <span class="candidate-count">${jobData.candidates.length} candidate${jobData.candidates.length !== 1 ? 's' : ''}</span>
                </div>
            </div>
            ${isLSPUJob ? this.renderLSPUJobDetails(jobData) : this.renderBasicJobDetails(jobData)}
            <div class="candidates-table-container">
                <div class="table-responsive">
                    <table class="table table-hover candidates-table-compact">
                        <thead class="table-dark">
                            <tr>
                                <th class="checkbox-header">
                                    <input type="checkbox" class="select-all-candidates" title="Select All">
                                </th>
                                <th class="candidate-header">Candidate</th>
                                <th class="gov-ids-header">Government IDs</th>
                                <th class="education-level-header">Education Level</th>
                                <th class="civil-service-header">Civil Service Eligibility</th>
                                <th class="score-header">Assessment Score</th>
                                <th class="status-header">Status</th>
                                <th class="actions-header">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${this.renderCandidateRows(jobData.candidates)}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        return jobSection;
    },

    // Render LSPU job details with enhanced information
    renderLSPUJobDetails(jobData) {
        return `
            <div class="job-description lspu-job-details">
                <div class="row">
                    <div class="col-md-6">
                        <div class="job-detail-item">
                            <strong><i class="fas fa-building me-2"></i>Department:</strong> 
                            ${DOMUtils.escapeHtml(jobData.department_office || 'Not specified')}
                        </div>
                        <div class="job-detail-item">
                            <strong><i class="fas fa-map-marker-alt me-2"></i>Campus:</strong> 
                            ${DOMUtils.escapeHtml(jobData.campus_name)}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="job-detail-item">
                            <strong><i class="fas fa-money-bill-wave me-2"></i>Salary Grade:</strong> 
                            ${DOMUtils.escapeHtml(jobData.salary_grade || 'Not specified')}
                        </div>
                        <div class="job-detail-item">
                            <strong><i class="fas fa-tag me-2"></i>Position Type:</strong> 
                            ${DOMUtils.escapeHtml(jobData.position_category)}
                        </div>
                    </div>
                </div>
                ${jobData.job_description ? `
                    <div class="job-description-text mt-3">
                        <strong><i class="fas fa-info-circle me-2"></i>Description:</strong>
                        <p>${FormatUtils.truncateText(jobData.job_description, 300)}</p>
                    </div>
                ` : ''}
                ${jobData.job_requirements ? `
                    <div class="job-requirements mt-2">
                        <strong><i class="fas fa-list-check me-2"></i>Required Skills:</strong> 
                        ${DOMUtils.escapeHtml(jobData.job_requirements)}
                    </div>
                ` : ''}
            </div>
        `;
    },

    // Render basic job details for legacy/unassigned categories
    renderBasicJobDetails(jobData) {
        if (!jobData.job_description && !jobData.job_requirements) {
            return `
                <div class="job-description">
                    <p class="text-muted"><i class="fas fa-info-circle me-2"></i>Candidates not yet assigned to a specific LSPU job posting.</p>
                </div>
            `;
        }
        
        return `
            <div class="job-description">
                ${jobData.job_description ? `<p>${FormatUtils.truncateText(jobData.job_description, 200)}</p>` : ''}
                ${jobData.job_requirements ? `
                    <div class="job-requirements">
                        <strong>Required Skills:</strong> ${DOMUtils.escapeHtml(jobData.job_requirements)}
                    </div>
                ` : ''}
            </div>
        `;
    },

    // Render candidate table rows
    renderCandidateRows(candidates) {
        return candidates.map(candidate => {
            const assessmentScore = candidate.assessment_score || candidate.score || 0;
            const scoreClass = this.getScoreColorClass(assessmentScore);
            const statusClass = `status-${candidate.status.toLowerCase()}`;
            const isSelected = this.selectedCandidates.has(candidate.id);
            const processingTypeLabel = this.getProcessingTypeLabel(candidate.processing_type, candidate.ocr_confidence);
            
            // Extract PDS-specific data with fallbacks
            const governmentIds = this.formatGovernmentIds(candidate);
            const educationLevel = this.getHighestEducationLevel(candidate);
            const civilServiceEligibility = this.formatCivilServiceEligibility(candidate);
            const assessmentScoreFormatted = this.formatAssessmentScore(candidate);
            
            return `
                <tr data-candidate-id="${candidate.id}" class="candidate-row ${isSelected ? 'selected' : ''}" onclick="CandidatesModule.showCandidateDetails('${candidate.id}')">
                    <td class="checkbox-column">
                        <input type="checkbox" class="candidate-checkbox" 
                               ${isSelected ? 'checked' : ''} 
                               data-candidate-id="${candidate.id}"
                               onclick="event.stopPropagation()">
                    </td>
                    <td class="candidate-column">
                        <div class="candidate-compact">
                            <div class="candidate-avatar">
                                <i class="fas fa-user-circle"></i>
                            </div>
                            <div class="candidate-info">
                                <div class="candidate-name">${DOMUtils.escapeHtml(candidate.name)}</div>
                                <div class="candidate-meta">
                                    <span class="candidate-email">${DOMUtils.escapeHtml(candidate.email)}</span>
                                    <span class="candidate-phone">${DOMUtils.escapeHtml(candidate.phone || 'No phone')}</span>
                                </div>
                                <div class="candidate-education">
                                    ${FormatUtils.truncateText(candidate.education, 60)}
                                </div>
                                <div class="processing-type-label">
                                    ${processingTypeLabel}
                                </div>
                            </div>
                        </div>
                    </td>
                    <td class="gov-ids-column">
                        <div class="gov-ids-compact">
                            ${governmentIds}
                        </div>
                    </td>
                    <td class="education-level-column">
                        <div class="education-level-compact">
                            <span class="education-badge">${educationLevel}</span>
                        </div>
                    </td>
                    <td class="civil-service-column">
                        <div class="civil-service-compact">
                            ${civilServiceEligibility}
                        </div>
                    </td>
                    <td class="score-column">
                        <div class="score-compact">
                            <span class="score-badge ${scoreClass}">${assessmentScoreFormatted}</span>
                            <div class="score-bar-mini">
                                <div class="score-fill ${scoreClass}" style="width: ${assessmentScore}%"></div>
                            </div>
                        </div>
                    </td>
                    <td class="status-column">
                        <span class="status-badge ${statusClass}">${candidate.status}</span>
                    </td>
                    <td class="actions-column">
                        <div class="action-buttons-compact">
                            <button class="btn btn-sm btn-outline-success shortlist-candidate" 
                                    title="Shortlist" onclick="event.stopPropagation(); CandidatesModule.updateCandidateStatus('${candidate.id}', 'shortlisted')">
                                <i class="fas fa-star"></i>
                            </button>
                            <button class="btn btn-sm btn-outline-danger reject-candidate" 
                                    title="Reject" onclick="event.stopPropagation(); CandidatesModule.updateCandidateStatus('${candidate.id}', 'rejected')">
                                <i class="fas fa-times"></i>
                            </button>
                            <div class="btn-group">
                                <button class="btn btn-sm btn-outline-secondary dropdown-toggle" 
                                        data-bs-toggle="dropdown" title="More" onclick="event.stopPropagation()">
                                    <i class="fas fa-ellipsis-v"></i>
                                </button>
                                <ul class="dropdown-menu">
                                    <li><button class="dropdown-item view-candidate" onclick="CandidatesModule.showCandidateDetails('${candidate.id}')">
                                        <i class="fas fa-eye me-2"></i>View Details</button></li>
                                    <li><button class="dropdown-item shortlist-candidate" onclick="CandidatesModule.updateCandidateStatus('${candidate.id}', 'pending')">
                                        <i class="fas fa-clock me-2"></i>Set Pending</button></li>
                                    <li><hr class="dropdown-divider"></li>
                                    <li><button class="dropdown-item remove-candidate text-danger" onclick="CandidatesModule.handleRemoveCandidate('${candidate.id}')">
                                        <i class="fas fa-trash me-2"></i>Remove</button></li>
                                </ul>
                            </div>
                        </div>
                    </td>
                </tr>
            `;
        }).join('');
    },

    // Setup candidate action listeners
    setupCandidateActionListeners() {
        if (!this.candidatesContent) return;

        // Individual candidate checkboxes
        this.candidatesContent.addEventListener('change', (e) => {
            if (e.target.classList.contains('candidate-checkbox')) {
                const candidateId = e.target.dataset.candidateId;
                const candidateRow = e.target.closest('.candidate-row');
                
                if (e.target.checked) {
                    this.selectedCandidates.add(candidateId);
                    candidateRow.classList.add('selected');
                } else {
                    this.selectedCandidates.delete(candidateId);
                    candidateRow.classList.remove('selected');
                }
                
                this.updateBulkActionsVisibility();
                this.updateSelectAllState();
            }
        });

        // Select all checkboxes
        this.candidatesContent.addEventListener('change', (e) => {
            if (e.target.classList.contains('select-all-candidates')) {
                const table = e.target.closest('table');
                const checkboxes = table.querySelectorAll('.candidate-checkbox');
                const isChecked = e.target.checked;
                
                checkboxes.forEach(checkbox => {
                    const candidateId = checkbox.dataset.candidateId;
                    const candidateRow = checkbox.closest('.candidate-row');
                    
                    checkbox.checked = isChecked;
                    
                    if (isChecked) {
                        this.selectedCandidates.add(candidateId);
                        candidateRow.classList.add('selected');
                    } else {
                        this.selectedCandidates.delete(candidateId);
                        candidateRow.classList.remove('selected');
                    }
                });
                
                this.updateBulkActionsVisibility();
            }
        });

        // Candidate actions
        this.candidatesContent.addEventListener('click', async (e) => {
            const candidateRow = e.target.closest('.candidate-row');
            if (!candidateRow) return;
            
            const candidateId = candidateRow.dataset.candidateId;
            
            // Prevent row click when interacting with controls
            if (e.target.closest('.candidate-checkbox') || 
                e.target.closest('.action-buttons-compact') ||
                e.target.closest('.dropdown-menu')) {
                return;
            }
            
            // Handle button clicks
            if (e.target.closest('.view-candidate')) {
                e.stopPropagation();
                await this.showCandidateDetails(candidateId);
            } else if (e.target.closest('.shortlist-candidate')) {
                e.stopPropagation();
                await this.updateCandidateStatus(candidateId, 'shortlisted');
            } else if (e.target.closest('.reject-candidate')) {
                e.stopPropagation();
                await this.updateCandidateStatus(candidateId, 'rejected');
            } else if (e.target.closest('.remove-candidate')) {
                e.stopPropagation();
                const confirmed = await confirmRemove('this candidate');
                if (confirmed) {
                    await this.removeCandidate(candidateId);
                }
            }
            // Row click is handled by onclick attribute in the HTML for better performance
        });
    },

    // Update bulk actions visibility
    updateBulkActionsVisibility() {
        const bulkActionsContainer = document.getElementById('bulkActionsContainer');
        const selectedCount = this.selectedCandidates.size;
        
        if (bulkActionsContainer) {
            if (selectedCount > 0) {
                bulkActionsContainer.style.display = 'block';
                bulkActionsContainer.querySelector('.selected-count').textContent = selectedCount;
            } else {
                bulkActionsContainer.style.display = 'none';
            }
        }
    },

    // Update select all checkbox state
    updateSelectAllState() {
        const selectAllCheckboxes = this.candidatesContent.querySelectorAll('.select-all-candidates');
        
        selectAllCheckboxes.forEach(selectAll => {
            const table = selectAll.closest('table');
            const allCheckboxes = table.querySelectorAll('.candidate-checkbox');
            const checkedCheckboxes = table.querySelectorAll('.candidate-checkbox:checked');
            
            if (checkedCheckboxes.length === 0) {
                selectAll.indeterminate = false;
                selectAll.checked = false;
            } else if (checkedCheckboxes.length === allCheckboxes.length) {
                selectAll.indeterminate = false;
                selectAll.checked = true;
            } else {
                selectAll.indeterminate = true;
                selectAll.checked = false;
            }
        });
    },

    // Show bulk actions menu
    showBulkActionsMenu() {
        if (this.selectedCandidates.size === 0) {
            ToastUtils.showWarning('Please select candidates first');
            return;
        }

        // Create bulk actions modal or dropdown
        const actions = [
            { id: 'bulk-shortlist', label: 'Shortlist Selected', icon: 'fas fa-star', action: () => this.bulkUpdateStatus('shortlisted') },
            { id: 'bulk-reject', label: 'Reject Selected', icon: 'fas fa-times', action: () => this.bulkUpdateStatus('rejected') },
            { id: 'bulk-pending', label: 'Set as Pending', icon: 'fas fa-clock', action: () => this.bulkUpdateStatus('pending') },
            { id: 'bulk-remove', label: 'Remove Selected', icon: 'fas fa-trash', action: () => this.bulkRemoveCandidates(), className: 'text-danger' }
        ];

        // You can implement a proper modal here or use a simple confirm approach
        this.showBulkActionsDialog(actions);
    },

    // Show bulk actions dialog
    showBulkActionsDialog(actions) {
        const selectedCount = this.selectedCandidates.size;
        let actionsHtml = actions.map(action => 
            `<button class="dropdown-item ${action.className || ''}" data-action="${action.id}">
                <i class="${action.icon} me-2"></i>${action.label}
            </button>`
        ).join('');

        // Simple implementation using browser confirm - you can enhance this with a proper modal
        const actionChoice = prompt(`Selected ${selectedCount} candidates. Choose action:\n1. Shortlist\n2. Reject\n3. Set as Pending\n4. Remove\n\nEnter number (1-4):`);
        
        switch(actionChoice) {
            case '1':
                this.bulkUpdateStatus('shortlisted');
                break;
            case '2':
                this.bulkUpdateStatus('rejected');
                break;
            case '3':
                this.bulkUpdateStatus('pending');
                break;
            case '4':
                this.bulkRemoveCandidates();
                break;
        }
    },

    // Bulk update candidate status
    async bulkUpdateStatus(status) {
        const selectedIds = Array.from(this.selectedCandidates);
        const updatePromises = selectedIds.map(id => this.updateCandidateStatus(id, status, false));
        
        try {
            await Promise.all(updatePromises);
            ToastUtils.showSuccess(`${selectedIds.length} candidates updated to ${status}`);
            this.selectedCandidates.clear();
            this.updateBulkActionsVisibility();
            await this.loadCandidates();
        } catch (error) {
            ToastUtils.showError('Some candidates could not be updated');
        }
    },

    // Bulk remove candidates
    async bulkRemoveCandidates() {
        const selectedIds = Array.from(this.selectedCandidates);
        const confirmed = await confirmRemove(`${selectedIds.length} candidates`);
        
        if (!confirmed) return;

        const removePromises = selectedIds.map(id => this.removeCandidate(id, false));
        
        try {
            await Promise.all(removePromises);
            ToastUtils.showSuccess(`${selectedIds.length} candidates removed`);
            this.selectedCandidates.clear();
            this.updateBulkActionsVisibility();
            await this.loadCandidates();
        } catch (error) {
            ToastUtils.showError('Some candidates could not be removed');
        }
    },

    // Export candidates
    async exportCandidates() {
        try {
            ToastUtils.showInfo('Preparing export...');
            
            // Prepare export data
            const exportData = [];
            Object.values(this.candidatesData || {}).forEach(jobData => {
                jobData.candidates.forEach(candidate => {
                    // Handle LSPU job structure vs legacy/unassigned
                    const isLSPUJob = jobData.position_title && jobData.campus_name;
                    
                    const exportRow = {
                        name: candidate.name,
                        email: candidate.email,
                        phone: candidate.phone || '',
                        education: candidate.education,
                        skills: candidate.all_skills.join(', '),
                        predicted_category: candidate.predicted_category,
                        match_score: candidate.score,
                        status: candidate.status
                    };
                    
                    // Add LSPU-specific fields or legacy fields
                    if (isLSPUJob) {
                        exportRow.position_title = jobData.position_title;
                        exportRow.position_category = jobData.position_category;
                        exportRow.campus_name = jobData.campus_name;
                        exportRow.department_office = jobData.department_office || '';
                        exportRow.salary_grade = jobData.salary_grade || '';
                    } else {
                        exportRow.job_title = jobData.job_title || 'Unassigned';
                        exportRow.job_category = jobData.job_category || 'General';
                        exportRow.campus_name = '';
                        exportRow.department_office = '';
                        exportRow.salary_grade = '';
                    }
                    
                    exportData.push(exportRow);
                });
            });

            // Convert to CSV
            const csvContent = this.arrayToCSV(exportData);
            
            // Download file
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', `candidates_export_${new Date().toISOString().split('T')[0]}.csv`);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            ToastUtils.showSuccess('Candidates exported successfully');
        } catch (error) {
            console.error('Export error:', error);
            ToastUtils.showError('Failed to export candidates');
        }
    },

    // Clear all filters
    clearFilters() {
        if (this.searchInput) this.searchInput.value = '';
        if (this.sortSelect) this.sortSelect.value = 'score-desc';
        if (this.filterSelect) this.filterSelect.value = 'all';
        this.selectedCandidates.clear();
        this.updateBulkActionsVisibility();
        this.filterAndDisplayCandidates();
    },

    // Convert array to CSV
    arrayToCSV(data) {
        if (!data.length) return '';
        
        const headers = Object.keys(data[0]);
        const csvRows = [];
        
        // Add header row
        csvRows.push(headers.map(header => `"${header}"`).join(','));
        
        // Add data rows
        data.forEach(row => {
            const values = headers.map(header => {
                const value = row[header] || '';
                return `"${String(value).replace(/"/g, '""')}"`;
            });
            csvRows.push(values.join(','));
        });
        
        return csvRows.join('\n');
    },

    // Show candidate details modal
    async showCandidateDetails(candidateId) {
        if (!this.modal) return;

        try {
            const data = await APIService.candidates.getById(candidateId);
            
            if (data.success) {
                const candidate = data.candidate;
                this.populateModal(candidate);
                this.modal.show();
            } else {
                ToastUtils.showError('Failed to load candidate details');
            }
        } catch (error) {
            console.error('Error loading candidate details:', error);
            ToastUtils.showError('Error loading candidate details');
        }
    },

    // Populate modal with candidate data
    populateModal(candidate) {
        // Basic info
        document.querySelector('#candidateDetailsModal .candidate-name').textContent = candidate.name;
        document.querySelector('#candidateDetailsModal .email').textContent = candidate.email || 'N/A';
        document.querySelector('#candidateDetailsModal .phone').textContent = candidate.phone || 'N/A';
        
        // Initialize score circle with loading state
        const scoreCircle = document.querySelector('#candidateDetailsModal .score-circle');
        scoreCircle.className = `score-circle score-loading`;
        scoreCircle.querySelector('.score-value').textContent = '...';
        
        // Fetch assessment data to show actual assessment score instead of match score
        this.fetchAssessmentData(candidate.id).then(assessmentData => {
            if (assessmentData) {
                const overallTotal = assessmentData.overall_total || 0;
                scoreCircle.className = `score-circle ${this.getScoreColorClass(overallTotal)}`;
                scoreCircle.querySelector('.score-value').textContent = `${overallTotal}`;
            } else {
                // Fallback to match score if assessment not available
                scoreCircle.className = `score-circle ${this.getScoreColorClass(candidate.matchScore)}`;
                scoreCircle.querySelector('.score-value').textContent = `${candidate.matchScore}%`;
            }
        }).catch(error => {
            console.error('Error fetching assessment score:', error);
            // Fallback to match score if error
            scoreCircle.className = `score-circle ${this.getScoreColorClass(candidate.matchScore)}`;
            scoreCircle.querySelector('.score-value').textContent = `${candidate.matchScore}%`;
        });
        
        // Check if this is a PDS candidate (legacy or new comprehensive system)
        console.log('Candidate data:', candidate); // Debug log
        console.log('Processing type:', candidate.processing_type); // Debug log
        console.log('PDS data exists:', !!candidate.pds_data); // Debug log
        
        const isPDS = candidate.processing_type === 'pds' || 
                     candidate.processing_type === 'comprehensive_pds_extraction' ||
                     candidate.processing_type === 'pds_extraction_fallback' ||
                     (candidate.pds_data && Object.keys(candidate.pds_data).length > 0);
        const pdsSection = document.querySelector('#candidateDetailsModal .pds-sections');
        
        console.log('Is PDS candidate:', isPDS); // Debug log
        
        // Hide/show sections based on candidate type
        if (isPDS) {
            pdsSection.style.display = 'block';
            this.populatePDSData(candidate);
            
            // Hide legacy resume sections for PDS candidates
            this.hideLegacySections();
        } else {
            pdsSection.style.display = 'none';
            
            // Show legacy resume sections for regular candidates
            this.showLegacySections();
            
            // Populate legacy sections
            this.populateLegacySections(candidate);
        }
        
        // Set up action buttons (common for both types)
        this.setupActionButtons(candidate);
    },

    // Hide legacy resume sections for PDS candidates
    hideLegacySections() {
        const legacySections = [
            '.skills-section',
            '.education-section', 
            '.experience-section',
            '.certifications-section',
            '.scoring-section',
            '.matched-skills-section',
            '.missing-skills-section'
        ];
        
        legacySections.forEach(selector => {
            const section = document.querySelector(`#candidateDetailsModal ${selector}`);
            if (section) {
                section.style.display = 'none';
            }
        });
    },

    // Show legacy resume sections for regular candidates
    showLegacySections() {
        const legacySections = [
            '.skills-section',
            '.education-section', 
            '.experience-section',
            '.certifications-section',
            '.scoring-section',
            '.matched-skills-section',
            '.missing-skills-section'
        ];
        
        legacySections.forEach(selector => {
            const section = document.querySelector(`#candidateDetailsModal ${selector}`);
            if (section) {
                section.style.display = 'block';
            }
        });
    },

    // Populate legacy sections for regular candidates
    populateLegacySections(candidate) {
        const skillsContainer = document.querySelector('#candidateDetailsModal .skills-container');
        if (candidate.skills && candidate.skills.length > 0) {
            skillsContainer.innerHTML = candidate.skills.map(skill => 
                `<span class="skill-badge">${DOMUtils.escapeHtml(skill)}</span>`
            ).join('');
        } else {
            skillsContainer.innerHTML = '<p>No skills information available</p>';
        }
        
        // Education (for non-PDS candidates or additional education info)
        const educationContainer = document.querySelector('#candidateDetailsModal .education-container');
        if (candidate.education && candidate.education.length > 0) {
            educationContainer.innerHTML = candidate.education.map(edu => `
                <div class="education-item">
                    <h6>${DOMUtils.escapeHtml(edu.degree || 'Unknown Degree')}</h6>
                    <p class="text-muted">${DOMUtils.escapeHtml(edu.year || 'Year not specified')}</p>
                    <p>${DOMUtils.escapeHtml(edu.details || '')}</p>
                </div>
            `).join('');
        } else {
            educationContainer.innerHTML = '<p>No education information available</p>';
        }
        
        // Matched skills
        const matchedSkillsContainer = document.querySelector('#candidateDetailsModal .matched-skills-container');
        if (candidate.matched_skills && candidate.matched_skills.length > 0) {
            matchedSkillsContainer.innerHTML = candidate.matched_skills.map(skill => 
                `<span class="skill-badge bg-success">${DOMUtils.escapeHtml(skill)}</span>`
            ).join('');
        } else {
            matchedSkillsContainer.innerHTML = '<p>No matched skills</p>';
        }
        
        // Missing skills
        const missingSkillsContainer = document.querySelector('#candidateDetailsModal .missing-skills-container');
        if (candidate.missing_skills && candidate.missing_skills.length > 0) {
            missingSkillsContainer.innerHTML = candidate.missing_skills.map(skill => 
                `<span class="skill-badge bg-danger">${DOMUtils.escapeHtml(skill)}</span>`
            ).join('');
        } else {
            missingSkillsContainer.innerHTML = '<p>No missing skills</p>';
        }
    },

    // Set up modal action buttons (common for both PDS and regular candidates)
    setupActionButtons(candidate) {
        const candidateId = candidate.id;
        document.getElementById('removeCandidate').dataset.candidateId = candidateId;
        document.getElementById('shortlistCandidate').dataset.candidateId = candidateId;
        document.getElementById('rejectCandidate').dataset.candidateId = candidateId;
    },

    // Populate PDS-specific data sections
    populatePDSData(candidate) {
        console.log('Starting PDS data population for candidate:', candidate);
        const pdsData = candidate.pds_data || {};
        console.log('PDS Data:', pdsData);
        
        // Personal Information
        this.populatePersonalInfo(pdsData);
        
        // Educational Background (new PDS section)
        this.populateEducationalBackground(candidate, pdsData);
        
        // Government IDs
        const govIdsContainer = document.querySelector('#candidateDetailsModal .government-ids-container');
        let govIds = candidate.government_ids || {};
        
        // If government_ids is empty, try to extract from PDS data
        if (Object.keys(govIds).length === 0 && pdsData.personal_info) {
            const personalInfo = pdsData.personal_info;
            govIds = {
                gsis_id: personalInfo.gsis_id,
                pagibig_id: personalInfo.pagibig_id,
                philhealth_no: personalInfo.philhealth_no,
                sss_no: personalInfo.sss_no,
                tin_no: personalInfo.tin_no
            };
        }
        
        const validGovIds = Object.entries(govIds)
            .filter(([key, value]) => value && value.trim() !== '' && value.toLowerCase() !== 'n/a');
            
        if (validGovIds.length > 0) {
            govIdsContainer.innerHTML = validGovIds.map(([key, value]) => `
                <div class="id-item">
                    <strong>${this.formatIDLabel(key)}:</strong> ${DOMUtils.escapeHtml(value)}
                </div>
            `).join('');
        } else {
            govIdsContainer.innerHTML = '<p>No government ID information available</p>';
        }
        
        // Civil Service Eligibility
        const eligibilityContainer = document.querySelector('#candidateDetailsModal .eligibility-container');
        const eligibility = candidate.eligibility || [];
        if (eligibility.length > 0) {
            const validEligibility = eligibility.filter(elig => 
                elig.eligibility && 
                elig.eligibility.trim() !== '' && 
                !elig.eligibility.includes('WORK EXPERIENCE') &&
                !elig.eligibility.includes('Continue on separate')
            );
            
            if (validEligibility.length > 0) {
                eligibilityContainer.innerHTML = validEligibility.map(elig => `
                    <div class="eligibility-item">
                        <h6>${DOMUtils.escapeHtml(elig.eligibility)}</h6>
                        <p class="text-muted">
                            ${elig.rating ? `Rating: ${elig.rating}` : ''} 
                            ${elig.date_exam ? `| Date: ${elig.date_exam}` : ''} 
                            ${elig.place_exam ? `| Place: ${elig.place_exam}` : ''}
                        </p>
                        ${elig.license_no ? `<p>License: ${DOMUtils.escapeHtml(elig.license_no)}</p>` : ''}
                        ${elig.validity ? `<p>Validity: ${DOMUtils.escapeHtml(elig.validity)}</p>` : ''}
                    </div>
                `).join('');
            } else {
                eligibilityContainer.innerHTML = '<p>No civil service eligibility information available</p>';
            }
        } else {
            eligibilityContainer.innerHTML = '<p>No civil service eligibility information available</p>';
        }
        
        // Work Experience (PDS)
        const workExpContainer = document.querySelector('#candidateDetailsModal .work-experience-container');
        const workExperience = candidate.work_experience || candidate.experience || [];
        if (workExperience.length > 0) {
            const validWorkExp = workExperience.filter(work => 
                work.position && 
                work.position.trim() !== '' && 
                work.position !== 'To' &&
                work.company && 
                work.company.trim() !== ''
            );
            
            if (validWorkExp.length > 0) {
                workExpContainer.innerHTML = validWorkExp.map(work => `
                    <div class="work-experience-item">
                        <h6>${DOMUtils.escapeHtml(work.position)}</h6>
                        <div class="company">${DOMUtils.escapeHtml(work.company)}</div>
                        <div class="date-range">
                            ${work.date_from ? new Date(work.date_from).toLocaleDateString() : 'N/A'} - 
                            ${work.date_to ? new Date(work.date_to).toLocaleDateString() : 'Present'}
                        </div>
                        ${work.status ? `<div class="description">${DOMUtils.escapeHtml(work.status)}</div>` : ''}
                        ${work.salary ? `<div class="text-muted">Salary: ${DOMUtils.escapeHtml(work.salary)}</div>` : ''}
                        ${work.govt_service ? `<div class="text-muted">Government Service: ${work.govt_service}</div>` : ''}
                    </div>
                `).join('');
            } else {
                workExpContainer.innerHTML = '<p>No work experience information available</p>';
            }
        } else {
            workExpContainer.innerHTML = '<p>No work experience information available</p>';
        }
        
        // Training and Development
        const trainingContainer = document.querySelector('#candidateDetailsModal .training-container');
        const training = candidate.training || [];
        if (training.length > 0) {
            const validTraining = training.filter(train => 
                train.title && 
                train.title.trim() !== '' && 
                train.title !== 'From'
            );
            
            if (validTraining.length > 0) {
                trainingContainer.innerHTML = validTraining.map(train => `
                    <div class="training-item">
                        <h6>${DOMUtils.escapeHtml(train.title)}</h6>
                        <p class="text-muted">
                            ${train.date_from || train.type ? 
                                `${train.type || train.date_from || ''} ${train.conductor ? `to ${train.conductor}` : ''}` : 
                                'Dates not specified'
                            }
                            ${train.hours ? `| ${train.hours} hours` : ''}
                        </p>
                    </div>
                `).join('');
            } else {
                trainingContainer.innerHTML = '<p>No training information available</p>';
            }
        } else {
            trainingContainer.innerHTML = '<p>No training information available</p>';
        }
        
        // Volunteer Work
        const volunteerContainer = document.querySelector('#candidateDetailsModal .volunteer-container');
        const volunteerWork = candidate.voluntary_work || candidate.volunteer_work || [];
        console.log('Volunteer work data:', volunteerWork); // Debug log
        
        if (volunteerWork.length > 0) {
            const validVolunteerWork = volunteerWork.filter(vol => 
                vol.organization && 
                vol.organization.trim() !== '' &&
                vol.organization !== 'From'
            );
            
            if (validVolunteerWork.length > 0) {
                volunteerContainer.innerHTML = validVolunteerWork.map(vol => `
                    <div class="volunteer-item">
                        <h6>${DOMUtils.escapeHtml(vol.organization)}</h6>
                        <p class="text-muted">
                            ${vol.date_from || vol.position ? 
                                `${vol.position || vol.date_from || ''}` : 
                                'Dates not specified'
                            }
                            ${vol.hours ? `| ${vol.hours} hours` : ''}
                        </p>
                    </div>
                `).join('');
            } else {
                volunteerContainer.innerHTML = '<p>No volunteer work information available</p>';
            }
        } else {
            volunteerContainer.innerHTML = '<p>No volunteer work information available</p>';
        }
        
        // Personal References
        const referencesContainer = document.querySelector('#candidateDetailsModal .references-container');
        const references = candidate.personal_references || (pdsData.other_info && pdsData.other_info.references) || [];
        if (references.length > 0) {
            const validReferences = references.filter(ref => 
                ref.name && 
                ref.name.trim() !== '' &&
                !ref.name.includes('42.') &&
                !ref.name.includes('declare under oath')
            );
            
            if (validReferences.length > 0) {
                referencesContainer.innerHTML = validReferences.map(ref => `
                    <div class="reference-item">
                        <h6>${DOMUtils.escapeHtml(ref.name)}</h6>
                        <p class="text-muted">
                            ${ref.address || 'N/A'} 
                            ${ref.telephone_no || ref.tel_no ? `| ${ref.telephone_no || ref.tel_no}` : ''}
                        </p>
                    </div>
                `).join('');
            } else {
                referencesContainer.innerHTML = '<p>No personal references available</p>';
            }
        } else {
            referencesContainer.innerHTML = '<p>No personal references available</p>';
        }
        
        // Assessment Results (new PDS section)
        this.populateAssessmentResults(candidate);
    },

    // Populate Educational Background section for PDS candidates
    populateEducationalBackground(candidate, pdsData) {
        const educationContainer = document.querySelector('#candidateDetailsModal .educational-background-container');
        const education = candidate.education || pdsData.educational_background || [];
        
        if (education.length > 0) {
            const validEducation = education.filter(edu => 
                edu.school && 
                edu.school.trim() !== '' &&
                edu.school !== 'From' &&
                !edu.school.includes('GRADUATE STUDIES') &&
                !edu.school.includes('VOCATIONAL')
            );
            
            if (validEducation.length > 0) {
                educationContainer.innerHTML = validEducation.map(edu => `
                    <div class="education-item">
                        <h6>${DOMUtils.escapeHtml(edu.level || 'Unknown Level')}</h6>
                        <div class="school">${DOMUtils.escapeHtml(edu.school)}</div>
                        <div class="degree">${DOMUtils.escapeHtml(edu.degree_course || edu.degree || '')}</div>
                        <div class="date-range text-muted">
                            ${edu.period_from ? edu.period_from : 'N/A'} - 
                            ${edu.period_to ? edu.period_to : (edu.year_graduated || 'N/A')}
                        </div>
                        ${edu.honors ? `<div class="honors"><i class="fas fa-award text-warning"></i> ${DOMUtils.escapeHtml(edu.honors)}</div>` : ''}
                        ${edu.highest_level_units ? `<div class="text-muted">Units: ${DOMUtils.escapeHtml(edu.highest_level_units)}</div>` : ''}
                    </div>
                `).join('');
            } else {
                educationContainer.innerHTML = '<p>No educational background information available</p>';
            }
        } else {
            educationContainer.innerHTML = '<p>No educational background information available</p>';
        }
    },

    // Populate Assessment Results section for PDS candidates
    populateAssessmentResults(candidate) {
        const assessmentContainer = document.querySelector('#candidateDetailsModal .assessment-results-container');
        
        // Show loading state
        assessmentContainer.innerHTML = `
            <div class="text-center p-4">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading assessment...</span>
                </div>
                <p class="mt-2">Calculating assessment results...</p>
            </div>
        `;
        
        // Fetch assessment data from API
        this.fetchAssessmentData(candidate.id).then(assessmentData => {
            if (assessmentData) {
                this.renderAssessmentResults(candidate, assessmentData);
            } else {
                this.renderNoAssessmentData();
            }
        }).catch(error => {
            console.error('Error fetching assessment data:', error);
            this.renderAssessmentError();
        });
    },

    async fetchAssessmentData(candidateId) {
        try {
            const response = await fetch(`/api/candidates/${candidateId}/assessment`);
            if (response.ok) {
                const result = await response.json();
                return result.success ? result.assessment : null;
            }
            return null;
        } catch (error) {
            console.error('Error fetching assessment data:', error);
            return null;
        }
    },

    renderAssessmentResults(candidate, assessmentData) {
        const assessmentContainer = document.querySelector('#candidateDetailsModal .assessment-results-container');
        
        // Extract scores from assessment data
        const breakdown = {
            education: assessmentData.education_score || 0,
            experience: assessmentData.experience_score || 0,
            training: assessmentData.training_score || 0,
            eligibility: assessmentData.eligibility_score || 0,
            accomplishments: assessmentData.accomplishments_score || 0,
            potential: assessmentData.potential_score || 0
        };
        
        const automatedTotal = assessmentData.automated_total || 0;
        const overallTotal = assessmentData.overall_total || 0;
        const percentageScore = (overallTotal / 100) * 100;
        
        assessmentContainer.innerHTML = `
            <div class="assessment-overview">
                <div class="assessment-scores-row">
                    <div class="score-section">
                        <div class="score-circle-large ${this.getScoreColorClass(automatedTotal)}">
                            <span class="score-value">${automatedTotal}</span>
                            <span class="score-label">Automated</span>
                        </div>
                        <p class="score-description">85 points maximum</p>
                    </div>
                    <div class="score-section">
                        <div class="score-circle-large ${this.getScoreColorClass(overallTotal)}">
                            <span class="score-value">${overallTotal}</span>
                            <span class="score-label">Overall</span>
                        </div>
                        <p class="score-description">100 points total</p>
                    </div>
                </div>
            </div>
            
            <div class="assessment-breakdown">
                <div class="criteria-header">
                    <h6>University Assessment Criteria</h6>
                    <small class="text-muted">Based on LSPU Standards</small>
                </div>
                
                <div class="criteria-list">
                    <div class="criteria-item">
                        <span class="criteria-label">I. Potential (15%) - Manual Entry</span>
                        <div class="criteria-controls">
                            <div class="potential-input-group">
                                <input type="number" 
                                       id="potentialScore" 
                                       class="form-control form-control-sm potential-input" 
                                       value="${breakdown.potential}" 
                                       min="0" 
                                       max="15" 
                                       step="0.1"
                                       data-candidate-id="${candidate.id}">
                                <span class="input-label">/ 15</span>
                                <button class="btn btn-sm btn-primary update-potential-btn" 
                                        onclick="CandidatesModule.updatePotentialScore(${candidate.id})">
                                    Update
                                </button>
                            </div>
                            <small class="text-muted">Interview (10%) + Aptitude Test (5%)</small>
                        </div>
                    </div>
                    
                    <div class="criteria-item automated">
                        <span class="criteria-label">II. Education (40%)</span>
                        <div class="criteria-bar">
                            <div class="criteria-fill education" style="width: ${(breakdown.education/40)*100}%"></div>
                        </div>
                        <span class="criteria-score">${breakdown.education}/40</span>
                    </div>
                    
                    <div class="criteria-item automated">
                        <span class="criteria-label">III. Experience (20%)</span>
                        <div class="criteria-bar">
                            <div class="criteria-fill experience" style="width: ${(breakdown.experience/20)*100}%"></div>
                        </div>
                        <span class="criteria-score">${breakdown.experience}/20</span>
                    </div>
                    
                    <div class="criteria-item automated">
                        <span class="criteria-label">IV. Training (10%)</span>
                        <div class="criteria-bar">
                            <div class="criteria-fill training" style="width: ${(breakdown.training/10)*100}%"></div>
                        </div>
                        <span class="criteria-score">${breakdown.training}/10</span>
                    </div>
                    
                    <div class="criteria-item automated">
                        <span class="criteria-label">V. Eligibility (10%)</span>
                        <div class="criteria-bar">
                            <div class="criteria-fill eligibility" style="width: ${(breakdown.eligibility/10)*100}%"></div>
                        </div>
                        <span class="criteria-score">${breakdown.eligibility}/10</span>
                    </div>
                    
                    <div class="criteria-item automated">
                        <span class="criteria-label">VI. Outstanding Accomplishments (5%)</span>
                        <div class="criteria-bar">
                            <div class="criteria-fill accomplishments" style="width: ${(breakdown.accomplishments/5)*100}%"></div>
                        </div>
                        <span class="criteria-score">${breakdown.accomplishments}/5</span>
                    </div>
                </div>
                
                <div class="assessment-summary">
                    <div class="summary-row">
                        <span class="summary-label">Automated Score (85%):</span>
                        <span class="summary-value">${automatedTotal}/85 points</span>
                    </div>
                    <div class="summary-row">
                        <span class="summary-label">Manual Score (15%):</span>
                        <span class="summary-value">${breakdown.potential}/15 points</span>
                    </div>
                    <div class="summary-row total">
                        <span class="summary-label">Total Score (100%):</span>
                        <span class="summary-value">${overallTotal}/100 points</span>
                    </div>
                    <div class="summary-row percentage">
                        <span class="summary-label">Percentage:</span>
                        <span class="summary-value">${percentageScore.toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;
    },

    renderNoAssessmentData() {
        const assessmentContainer = document.querySelector('#candidateDetailsModal .assessment-results-container');
        assessmentContainer.innerHTML = `
            <div class="text-center p-4">
                <p class="text-muted">No assessment data available for this candidate.</p>
                <p class="small">Assessment requires PDS data.</p>
            </div>
        `;
    },

    renderAssessmentError() {
        const assessmentContainer = document.querySelector('#candidateDetailsModal .assessment-results-container');
        assessmentContainer.innerHTML = `
            <div class="text-center p-4">
                <p class="text-danger">Error loading assessment data.</p>
                <button class="btn btn-sm btn-secondary" onclick="location.reload()">Refresh Page</button>
            </div>
        `;
    },
    // Update potential score via AJAX
    async updatePotentialScore(candidateId) {
        const input = document.getElementById('potentialScore');
        const updateBtn = document.querySelector('.update-potential-btn');
        const newScore = parseFloat(input.value) || 0;
        
        if (newScore < 0 || newScore > 15) {
            this.showNotification('Potential score must be between 0 and 15', 'error');
            input.focus();
            return;
        }
        
        // Show loading state
        const originalBtnText = updateBtn.textContent;
        updateBtn.disabled = true;
        updateBtn.textContent = 'Updating...';
        
        try {
            const response = await fetch('/api/update_potential_score', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    candidate_id: candidateId,
                    potential_score: newScore
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                
                if (result.success) {
                    // Update the assessment display immediately
                    this.updateAssessmentDisplay(candidateId, newScore);
                    
                    // Show success message
                    this.showNotification('Potential score updated successfully', 'success');
                    
                    // Note: We no longer need to call loadCandidates() since we update the row directly
                    // This provides faster feedback and better user experience
                } else {
                    throw new Error(result.error || 'Failed to update potential score');
                }
            } else {
                const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
            }
        } catch (error) {
            console.error('Error updating potential score:', error);
            this.showNotification(`Failed to update potential score: ${error.message}`, 'error');
        } finally {
            // Restore button state
            updateBtn.disabled = false;
            updateBtn.textContent = originalBtnText;
        }
    },
    
    // Update assessment display with new potential score
    updateAssessmentDisplay(candidateId, newPotentialScore) {
        // Refresh the assessment data directly from the API
        this.fetchAssessmentData(candidateId).then(assessmentData => {
            if (assessmentData) {
                // Update the assessment breakdown
                const candidate = { id: candidateId };
                this.renderAssessmentResults(candidate, assessmentData);
                
                // Update the top score circle with new overall score
                const overallTotal = assessmentData.overall_total || 0;
                const scoreCircle = document.querySelector('#candidateDetailsModal .score-circle');
                const scoreValue = scoreCircle.querySelector('.score-value');
                if (scoreValue) {
                    scoreValue.textContent = `${overallTotal}`;
                    scoreCircle.className = `score-circle ${this.getScoreColorClass(overallTotal)}`;
                }
                
                // Update the candidate's score in the main table
                this.updateCandidateRowScore(candidateId, overallTotal);
            }
        }).catch(error => {
            console.error('Error updating assessment display:', error);
        });
    },
    
    // Update candidate row score in the main table
    updateCandidateRowScore(candidateId, newAssessmentScore) {
        // Find the candidate row in the table
        const candidateRow = document.querySelector(`tr[data-candidate-id="${candidateId}"]`);
        if (candidateRow) {
            // Update the score column
            const scoreColumn = candidateRow.querySelector('.score-column');
            if (scoreColumn) {
                const scoreClass = this.getScoreColorClass(newAssessmentScore);
                scoreColumn.innerHTML = `
                    <div class="score-compact">
                        <span class="score-badge ${scoreClass}">${newAssessmentScore}/100</span>
                        <div class="score-bar-mini">
                            <div class="score-fill ${scoreClass}" style="width: ${newAssessmentScore}%"></div>
                        </div>
                    </div>
                `;
            }
        }
        
        // Also update the cached candidate data if it exists
        if (this.candidatesData) {
            // Find the candidate in the grouped data structure
            Object.values(this.candidatesData).forEach(jobData => {
                if (jobData.candidates) {
                    const candidate = jobData.candidates.find(c => c.id == candidateId);
                    if (candidate) {
                        candidate.assessment_score = newAssessmentScore;
                        candidate.score = newAssessmentScore;
                    }
                }
            });
        }
    },
    
    // Show notification message
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 3000);
    },

    // Populate personal information section
    populatePersonalInfo(pdsData) {
        const personalInfo = pdsData.personal_info || {};
        
        // Full Name
        const fullName = [
            personalInfo.first_name,
            personalInfo.middle_name,
            personalInfo.surname,
            personalInfo.name_extension
        ].filter(part => part && part.trim() !== '' && part.toLowerCase() !== 'n/a').join(' ');
        
        // Safe element access with null checks
        const setTextContent = (id, value) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value || 'N/A';
            } else {
                console.warn(`Element with ID '${id}' not found`);
            }
        };
        
        setTextContent('fullName', fullName);
        setTextContent('dateOfBirth', personalInfo.date_of_birth);
        setTextContent('placeOfBirth', personalInfo.place_of_birth);
        setTextContent('gender', personalInfo.sex);
        setTextContent('civilStatus', personalInfo.civil_status);
        setTextContent('citizenship', personalInfo.citizenship);
        
        // Physical Information
        setTextContent('height', personalInfo.height ? `${personalInfo.height} m` : null);
        setTextContent('weight', personalInfo.weight ? `${personalInfo.weight} kg` : null);
        setTextContent('bloodType', personalInfo.blood_type);
        
        // Contact Information
        setTextContent('mobileNo', personalInfo.mobile_no);
        setTextContent('telephoneNo', personalInfo.telephone_no);
        setTextContent('emailAddress', personalInfo.email);
        
        // Addresses
        const residentialAddr = personalInfo.residential_address ? 
            personalInfo.residential_address.full_address : null;
        const permanentAddr = personalInfo.permanent_address ? 
            personalInfo.permanent_address.full_address : null;
            
        setTextContent('residentialAddress', residentialAddr);
        setTextContent('permanentAddress', permanentAddr);
    },

    // Format ID labels for display
    formatIDLabel(key) {
        const labels = {
            'gsis_id': 'GSIS ID',
            'pagibig_id': 'Pag-IBIG ID',
            'philhealth_no': 'PhilHealth No.',
            'sss_no': 'SSS No.',
            'tin_no': 'TIN No.'
        };
        return labels[key] || key.replace('_', ' ').toUpperCase();
    },

    // Setup modal action buttons
    setupModalActions() {
        const removeBtn = document.getElementById('removeCandidate');
        const shortlistBtn = document.getElementById('shortlistCandidate');
        const rejectBtn = document.getElementById('rejectCandidate');
        
        if (removeBtn) {
            removeBtn.addEventListener('click', async () => {
                const candidateId = removeBtn.dataset.candidateId;
                const confirmed = await confirmRemove('this candidate');
                if (confirmed) {
                    await this.removeCandidate(candidateId);
                    this.modal.hide();
                }
            });
        }
        
        if (shortlistBtn) {
            shortlistBtn.addEventListener('click', async () => {
                const candidateId = shortlistBtn.dataset.candidateId;
                await this.updateCandidateStatus(candidateId, 'shortlisted');
                this.modal.hide();
            });
        }
        
        if (rejectBtn) {
            rejectBtn.addEventListener('click', async () => {
                const candidateId = rejectBtn.dataset.candidateId;
                await this.updateCandidateStatus(candidateId, 'rejected');
                this.modal.hide();
            });
        }
    },

    // Remove candidate
    async removeCandidate(candidateId, showToast = true) {
        try {
            const result = await APIService.candidates.delete(candidateId);
            
            if (result.success) {
                if (showToast) {
                    ToastUtils.showSuccess('Candidate removed successfully');
                    await this.loadCandidates();
                }
                return true;
            } else {
                if (showToast) {
                    ToastUtils.showError('Failed to remove candidate');
                }
                return false;
            }
        } catch (error) {
            console.error('Error removing candidate:', error);
            if (showToast) {
                ToastUtils.showError('Error removing candidate');
            }
            return false;
        }
    },

    // Update candidate status
    async updateCandidateStatus(candidateId, status, showToast = true) {
        try {
            const result = await APIService.candidates.updateStatus(candidateId, status);
            
            if (result.success) {
                if (showToast) {
                    ToastUtils.showSuccess(`Candidate ${status} successfully`);
                    await this.loadCandidates();
                }
                return true;
            } else {
                if (showToast) {
                    ToastUtils.showError('Failed to update candidate status');
                }
                return false;
            }
        } catch (error) {
            console.error('Error updating candidate status:', error);
            if (showToast) {
                ToastUtils.showError('Error updating candidate status');
            }
            return false;
        }
    },

    // Handle remove candidate with confirmation
    async handleRemoveCandidate(candidateId) {
        const confirmed = await confirmRemove('this candidate');
        if (confirmed) {
            await this.removeCandidate(candidateId);
        }
    },

    // Get processing type label with appropriate styling
    getProcessingTypeLabel(processingType, ocrConfidence = null) {
        const typeConfig = {
            'resume': {
                label: 'Resume',
                icon: 'fas fa-file-alt',
                class: 'processing-type-resume'
            },
            'pds': {
                label: 'PDS Excel',
                icon: 'fas fa-file-excel',
                class: 'processing-type-pds'
            },
            'pds_text': {
                label: 'PDS Text',
                icon: 'fas fa-file-text',
                class: 'processing-type-pds-text'
            },
            'pds_only': {
                label: 'PDS Only',
                icon: 'fas fa-id-card',
                class: 'processing-type-pds-only'
            },
            'ocr_scanned': {
                label: 'OCR Scanned',
                icon: 'fas fa-scanner',
                class: 'processing-type-ocr'
            }
        };

        const config = typeConfig[processingType] || {
            label: 'Unknown',
            icon: 'fas fa-question',
            class: 'processing-type-unknown'
        };

        // Add OCR confidence if available
        let confidenceDisplay = '';
        if (processingType === 'ocr_scanned' && ocrConfidence !== null && ocrConfidence !== undefined) {
            const confidenceClass = this.getConfidenceColorClass(ocrConfidence);
            confidenceDisplay = ` <span class="ocr-confidence-badge ${confidenceClass}" title="OCR Confidence: ${ocrConfidence}%">${Math.round(ocrConfidence)}%</span>`;
        }

        return `<span class="processing-type-badge ${config.class}" title="Processed using ${config.label}">
                    <i class="${config.icon}"></i> ${config.label}${confidenceDisplay}
                </span>`;
    },

    // Get score color class
    getScoreColorClass(score) {
        if (score >= 80) return 'score-excellent';
        if (score >= 60) return 'score-good';
        if (score >= 40) return 'score-fair';
        return 'score-poor';
    },

    // PDS-specific formatting methods for Phase 2 frontend modernization
    
    // Format government IDs for display
    formatGovernmentIds(candidate) {
        let govIds = candidate.government_ids || {};
        
        // If government_ids is empty, try to extract from PDS data
        if (Object.keys(govIds).length === 0 && candidate.pds_data && candidate.pds_data.personal_info) {
            const personalInfo = candidate.pds_data.personal_info;
            govIds = {
                gsis_id: personalInfo.gsis_id,
                pagibig_id: personalInfo.pagibig_id,
                philhealth_no: personalInfo.philhealth_no,
                sss_no: personalInfo.sss_no,
                tin_no: personalInfo.tin_no
            };
        }
        
        const ids = [];
        
        // Priority order for display - Updated to match actual PDS field names
        const idTypes = [
            { key: 'tin_no', label: 'TIN', icon: 'fa-id-card' },
            { key: 'sss_no', label: 'SSS', icon: 'fa-shield-alt' },
            { key: 'philhealth_no', label: 'PhilHealth', icon: 'fa-heartbeat' },
            { key: 'pagibig_id', label: 'Pag-IBIG', icon: 'fa-home' },
            { key: 'gsis_id', label: 'GSIS', icon: 'fa-university' }
        ];
        
        idTypes.forEach(idType => {
            const value = govIds[idType.key];
            if (value && 
                value.toString().trim() !== '' && 
                value.toString().toLowerCase() !== 'n/a' &&
                value.toString().toLowerCase() !== 'none' &&
                value.toString() !== 'null') {
                ids.push(`<span class="gov-id-item" title="${idType.label}: ${value}">
                    <i class="fas ${idType.icon}"></i> ${idType.label}
                </span>`);
            }
        });
        
        if (ids.length === 0) {
            return '<span class="text-muted"><i class="fas fa-id-card-alt"></i> Not provided</span>';
        }
        
        // Show max 2 IDs, with count if more
        const displayed = ids.slice(0, 2);
        const additional = ids.length > 2 ? `<span class="ids-count">+${ids.length - 2}</span>` : '';
        
        return displayed.join(' ') + additional;
    },
    
    // Get highest education level
    getHighestEducationLevel(candidate) {
        const education = candidate.education || [];
        if (!Array.isArray(education) || education.length === 0) {
            return '<span class="text-muted">Not specified</span>';
        }
        
        // Education level priority (highest to lowest) - Updated for PDS structure
        const levelPriority = {
            'graduate': 5,
            'doctoral': 5,
            'doctorate': 5,
            'phd': 5,
            'masters': 4,
            'master': 4,
            'college': 3,
            'bachelor': 3,
            'undergraduate': 3,
            'vocational': 2,
            'technical': 2,
            'trade': 2,
            'secondary': 1,
            'high school': 1,
            'elementary': 0
        };
        
        let highest = null;
        let highestPriority = -1;
        
        education.forEach(edu => {
            const level = (edu.level || '').toLowerCase();
            const degree = (edu.degree || edu.course || '').toLowerCase();
            const school = edu.school || edu.institution || '';
            
            // Check level first, then degree content
            let priority = levelPriority[level] || -1;
            
            if (priority === -1) {
                // Check degree content for keywords
                for (const [keyword, prio] of Object.entries(levelPriority)) {
                    if (degree.includes(keyword) && prio > priority) {
                        priority = prio;
                    }
                }
            }
            
            if (priority > highestPriority) {
                highest = edu;
                highestPriority = priority;
            }
        });
        
        if (highest) {
            const level = highest.level || 'Unknown';
            const school = highest.school || highest.institution || '';
            const displayText = level.charAt(0).toUpperCase() + level.slice(1);
            
            return `<span class="education-level" title="${displayText} - ${school}">
                <i class="fas fa-graduation-cap"></i> ${displayText}
            </span>`;
        }
        
        return '<span class="text-muted">Not classified</span>';
    },
    
    // Format civil service eligibility
    formatCivilServiceEligibility(candidate) {
        const eligibility = candidate.eligibility || [];
        if (!Array.isArray(eligibility) || eligibility.length === 0) {
            return '<span class="text-muted"><i class="fas fa-certificate"></i> None</span>';
        }
        
        // Filter out invalid entries and find the best eligibility
        const validEligibility = eligibility.filter(elig => 
            elig.eligibility && 
            elig.eligibility.trim() !== '' && 
            !elig.eligibility.includes('WORK EXPERIENCE') &&
            !elig.eligibility.includes('Continue on separate') &&
            !elig.eligibility.includes('28.') &&
            !elig.eligibility.includes('From') &&
            !elig.eligibility.includes('To')
        );
        
        if (validEligibility.length === 0) {
            return '<span class="text-muted"><i class="fas fa-certificate"></i> None</span>';
        }
        
        // Find the best eligibility (with rating or most recent)
        let best = null;
        let bestRating = 0;
        
        validEligibility.forEach(elig => {
            const rating = parseFloat(elig.rating || 0);
            const examName = elig.eligibility || '';
            
            if (examName && rating > bestRating) {
                best = elig;
                bestRating = rating;
            } else if (examName && !best) {
                best = elig; // Take first valid entry if no ratings found
            }
        });
        
        if (best) {
            const examType = best.eligibility || 'Civil Service';
            const rating = best.rating || '';
            
            let badgeClass = 'badge bg-secondary';
            if (rating && parseFloat(rating) >= 80) {
                badgeClass = 'badge bg-success';
            } else if (rating && parseFloat(rating) >= 70) {
                badgeClass = 'badge bg-warning';
            }
            
            const ratingText = rating ? ` (${rating}%)` : '';
            const title = `${examType}${ratingText}${best.date_exam ? ` - ${best.date_exam}` : ''}`;
            
            // Show count if multiple eligibilities
            const countText = validEligibility.length > 1 ? ` +${validEligibility.length - 1}` : '';
            
            return `<span class="${badgeClass}" title="${title}">
                <i class="fas fa-certificate"></i> ${FormatUtils.truncateText(examType, 12)}${ratingText}${countText}
            </span>`;
        }
        
        // Fallback: show count of eligibilities
        return `<span class="badge bg-info" title="${validEligibility.length} eligibility entries">
            <i class="fas fa-certificate"></i> ${validEligibility.length} entries
        </span>`;
    },
    
    // Format assessment score with breakdown
    formatAssessmentScore(candidate) {
        const assessmentScore = candidate.assessment_score || candidate.score || 0;
        
        // Show assessment score out of 100
        return `${assessmentScore}/100`;
    },

    // Get confidence color class for OCR confidence scores
    getConfidenceColorClass(confidence) {
        if (confidence >= 85) return 'confidence-high';
        if (confidence >= 70) return 'confidence-medium';
        if (confidence >= 50) return 'confidence-low';
        return 'confidence-very-low';
    }
};

// Make globally available
window.CandidatesModule = CandidatesModule;

// Backward compatibility
window.loadCandidatesSection = CandidatesModule.loadCandidates.bind(CandidatesModule);
