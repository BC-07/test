// Candidates Module
const CandidatesModule = {
    candidatesContent: null,
    modal: null,

    // Initialize candidates functionality
    init() {
        this.setupElements();
        this.setupEventListeners();
    },

    // Setup DOM elements
    setupElements() {
        this.candidatesContent = document.getElementById('candidatesContent');
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

        // Modal action buttons
        this.setupModalActions();
    },

    // Load candidates from API
    async loadCandidates() {
        if (!this.candidatesContent) return;

        try {
            const data = await APIService.candidates.getAll();
            
            if (data.success) {
                this.displayCandidatesByJob(data.candidates_by_job, data.total_candidates);
                this.setupCandidateActionListeners();
            } else {
                ToastUtils.showError('Failed to load candidates');
            }
        } catch (error) {
            console.error('Error loading candidates:', error);
            ToastUtils.showError('Error loading candidates');
        }
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
        jobSection.innerHTML = `
            <div class="job-header">
                <h3 class="job-title">
                    <i class="fas fa-briefcase me-2"></i>
                    ${DOMUtils.escapeHtml(jobData.job_title)}
                </h3>
                <div class="job-meta">
                    <span class="badge bg-primary">${DOMUtils.escapeHtml(jobData.job_category)}</span>
                    <span class="candidate-count">${jobData.candidates.length} candidate${jobData.candidates.length !== 1 ? 's' : ''}</span>
                </div>
            </div>
            <div class="job-description">
                <p>${FormatUtils.truncateText(jobData.job_description, 200)}</p>
                <div class="job-requirements">
                    <strong>Required Skills:</strong> ${DOMUtils.escapeHtml(jobData.job_requirements)}
                </div>
            </div>
            <div class="candidates-table-container">
                <div class="table-responsive">
                    <table class="table table-striped table-hover candidates-table">
                        <thead class="table-dark">
                            <tr>
                                <th>Name</th>
                                <th>Education</th>
                                <th>Phone</th>
                                <th>Skills (Top 5)</th>
                                <th>Predicted Category</th>
                                <th>Score</th>
                                <th>Status</th>
                                <th>Actions</th>
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

    // Render candidate table rows
    renderCandidateRows(candidates) {
        return candidates.map(candidate => {
            const scoreClass = this.getScoreColorClass(candidate.score);
            const statusClass = `status-${candidate.status.toLowerCase()}`;
            const topSkills = candidate.all_skills.slice(0, 5);
            
            return `
                <tr data-candidate-id="${candidate.id}" class="candidate-row">
                    <td>
                        <div class="candidate-name-cell">
                            <strong>${DOMUtils.escapeHtml(candidate.name)}</strong>
                            <small class="text-muted d-block">${DOMUtils.escapeHtml(candidate.email)}</small>
                        </div>
                    </td>
                    <td>
                        <span class="education-text" title="${DOMUtils.escapeHtml(candidate.education)}">
                            ${FormatUtils.truncateText(candidate.education, 50)}
                        </span>
                    </td>
                    <td>${DOMUtils.escapeHtml(candidate.phone || 'Not provided')}</td>
                    <td>
                        <div class="skills-cell">
                            ${FormatUtils.formatSkillTags(topSkills, 5, 'skill-tag small')}
                        </div>
                    </td>
                    <td>
                        <span class="predicted-category" title="AI Prediction: ${DOMUtils.escapeHtml(candidate.predicted_category)}">
                            ${DOMUtils.escapeHtml(candidate.predicted_category)}
                        </span>
                    </td>
                    <td>
                        <span class="score-badge ${scoreClass}">${candidate.score}%</span>
                    </td>
                    <td>
                        <span class="status-badge ${statusClass}">${candidate.status}</span>
                    </td>
                    <td>
                        <div class="action-buttons">
                            <button class="btn btn-sm btn-outline-primary view-candidate" title="View Details">
                                <i class="fas fa-eye"></i>
                            </button>
                            <button class="btn btn-sm btn-outline-success shortlist-candidate" title="Shortlist">
                                <i class="fas fa-star"></i>
                            </button>
                            <button class="btn btn-sm btn-outline-danger remove-candidate" title="Remove">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </td>
                </tr>
            `;
        }).join('');
    },

    // Setup candidate action listeners
    setupCandidateActionListeners() {
        if (!this.candidatesContent) return;

        this.candidatesContent.addEventListener('click', async (e) => {
            const candidateRow = e.target.closest('.candidate-row');
            if (!candidateRow) return;
            
            const candidateId = candidateRow.dataset.candidateId;
            
            if (e.target.closest('.view-candidate')) {
                e.stopPropagation();
                await this.showCandidateDetails(candidateId);
            } else if (e.target.closest('.shortlist-candidate')) {
                e.stopPropagation();
                await this.updateCandidateStatus(candidateId, 'shortlisted');
            } else if (e.target.closest('.remove-candidate')) {
                e.stopPropagation();
                if (confirm('Are you sure you want to remove this candidate?')) {
                    await this.removeCandidate(candidateId);
                }
            } else {
                // Row click - show details
                await this.showCandidateDetails(candidateId);
            }
        });
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
        
        // Score
        const scoreCircle = document.querySelector('#candidateDetailsModal .score-circle');
        scoreCircle.className = `score-circle ${this.getScoreColorClass(candidate.matchScore)}`;
        scoreCircle.querySelector('.score-value').textContent = `${candidate.matchScore}%`;
        
        // Skills
        const skillsContainer = document.querySelector('#candidateDetailsModal .skills-container');
        if (candidate.skills && candidate.skills.length > 0) {
            skillsContainer.innerHTML = candidate.skills.map(skill => 
                `<span class="skill-badge">${DOMUtils.escapeHtml(skill)}</span>`
            ).join('');
        } else {
            skillsContainer.innerHTML = '<p>No skills information available</p>';
        }
        
        // Education
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
        
        // Update action buttons
        const candidateId = candidate.id;
        document.getElementById('removeCandidate').dataset.candidateId = candidateId;
        document.getElementById('shortlistCandidate').dataset.candidateId = candidateId;
        document.getElementById('rejectCandidate').dataset.candidateId = candidateId;
    },

    // Setup modal action buttons
    setupModalActions() {
        const removeBtn = document.getElementById('removeCandidate');
        const shortlistBtn = document.getElementById('shortlistCandidate');
        const rejectBtn = document.getElementById('rejectCandidate');
        
        if (removeBtn) {
            removeBtn.addEventListener('click', async () => {
                const candidateId = removeBtn.dataset.candidateId;
                if (confirm('Are you sure you want to remove this candidate?')) {
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
    async removeCandidate(candidateId) {
        try {
            const result = await APIService.candidates.delete(candidateId);
            
            if (result.success) {
                ToastUtils.showSuccess('Candidate removed successfully');
                await this.loadCandidates();
            } else {
                ToastUtils.showError('Failed to remove candidate');
            }
        } catch (error) {
            console.error('Error removing candidate:', error);
            ToastUtils.showError('Error removing candidate');
        }
    },

    // Update candidate status
    async updateCandidateStatus(candidateId, status) {
        try {
            const result = await APIService.candidates.updateStatus(candidateId, status);
            
            if (result.success) {
                ToastUtils.showSuccess(`Candidate ${status} successfully`);
                await this.loadCandidates();
            } else {
                ToastUtils.showError('Failed to update candidate status');
            }
        } catch (error) {
            console.error('Error updating candidate status:', error);
            ToastUtils.showError('Error updating candidate status');
        }
    },

    // Get score color class
    getScoreColorClass(score) {
        if (score >= 80) return 'score-excellent';
        if (score >= 60) return 'score-good';
        if (score >= 40) return 'score-fair';
        return 'score-poor';
    }
};

// Make globally available
window.CandidatesModule = CandidatesModule;

// Backward compatibility
window.loadCandidatesSection = CandidatesModule.loadCandidates.bind(CandidatesModule);
