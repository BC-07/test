// Job Management Module
const JobsModule = {
    currentSkills: new Set(),
    jobsGrid: null,
    addJobBtn: null,
    jobForm: null,
    skillInput: null,
    skillTags: null,
    saveJobBtn: null,

    // Initialize job management functionality
    init() {
        this.setupElements();
        this.setupEventListeners();
        this.loadJobCategories();
        this.loadJobs();
    },

    // Setup DOM elements
    setupElements() {
        this.jobsGrid = document.getElementById('jobsGrid');
        this.addJobBtn = document.getElementById('addJobBtn');
        this.jobForm = document.getElementById('jobForm');
        this.skillInput = document.getElementById('skillInput');
        this.skillTags = document.getElementById('skillTags');
        this.saveJobBtn = document.getElementById('saveJobBtn');
    },

    // Setup event listeners
    setupEventListeners() {
        // Add job button
        if (this.addJobBtn) {
            this.addJobBtn.addEventListener('click', async () => {
                await this.showAddJobModal();
            });
        }

        // Skill input
        if (this.skillInput) {
            this.skillInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ',') {
                    e.preventDefault();
                    const skill = this.skillInput.value.trim();
                    if (skill) {
                        this.currentSkills.add(skill);
                        this.updateSkillTags();
                        this.skillInput.value = '';
                    }
                }
            });
        }

        // Form progress tracking
        const formFields = ['jobTitle', 'jobDepartment', 'jobCategory', 'jobExperience', 'jobDescription'];
        formFields.forEach(fieldId => {
            const field = document.getElementById(fieldId);
            if (field) {
                field.addEventListener('input', () => this.updateFormProgress());
                field.addEventListener('change', () => this.updateFormProgress());
            }
        });

        // Save job button
        if (this.saveJobBtn) {
            this.saveJobBtn.addEventListener('click', async (e) => {
                e.preventDefault();
                await this.saveJob();
            });
        }

        // Category management
        this.setupCategoryManagement();
    },

    // Load jobs from API
    async loadJobs() {
        try {
            const data = await APIService.jobs.getAll();
            
            if (!data.success) {
                throw new Error(data.error || 'Failed to load jobs');
            }
            
            const jobs = data.jobs || [];
            this.renderJobs(jobs);
        } catch (error) {
            console.error('Error loading jobs:', error);
            ToastUtils.showError(error.message || 'Failed to load jobs');
        }
    },

    // Render jobs in the grid
    renderJobs(jobs) {
        if (!this.jobsGrid) return;

        if (jobs.length === 0) {
            this.jobsGrid.innerHTML = `
                <div class="no-jobs-message">
                    <div class="no-jobs-icon">
                        <i class="fas fa-briefcase"></i>
                    </div>
                    <h4>No Jobs Available</h4>
                    <p>Create your first job posting to get started.</p>
                    <button class="btn btn-primary" onclick="JobsModule.showAddJobModal()">
                        <i class="fas fa-plus me-2"></i>Add Job
                    </button>
                </div>
            `;
            return;
        }

        this.jobsGrid.innerHTML = jobs.map(job => `
            <div class="job-card animate-fade-in-up" data-job-id="${job.id}">
                <div class="job-header">
                    <div>
                        <h3 class="job-title">${DOMUtils.escapeHtml(job.title)}</h3>
                        <div class="job-meta">
                            <span class="job-department">${DOMUtils.escapeHtml(job.department)}</span>
                        </div>
                    </div>
                    <div class="job-tags">
                        <span class="job-category badge bg-info">${DOMUtils.escapeHtml(job.category)}</span>
                        <span class="job-experience badge bg-secondary">${DOMUtils.escapeHtml(job.experience_level)}</span>
                    </div>
                </div>
                <div class="job-body">
                    <p class="job-description">${FormatUtils.truncateText(job.description, 150)}</p>
                    <div class="job-skills">
                        ${FormatUtils.formatSkillTags(job.requirements.split(',').map(s => s.trim()))}
                    </div>
                </div>
                <div class="job-footer">
                    <button class="btn btn-outline-primary btn-sm edit-job" data-job-id="${job.id}">
                        <i class="fas fa-edit me-1"></i>Edit
                    </button>
                    <button class="btn btn-outline-danger btn-sm delete-job" data-job-id="${job.id}">
                        <i class="fas fa-trash-alt me-1"></i>Delete
                    </button>
                </div>
            </div>
        `).join('');

        this.setupJobCardListeners();
    },

    // Setup job card event listeners
    setupJobCardListeners() {
        // Edit job buttons
        document.querySelectorAll('.edit-job').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.preventDefault();
                const jobId = btn.dataset.jobId;
                await this.editJob(jobId);
            });
        });

        // Delete job buttons
        document.querySelectorAll('.delete-job').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.preventDefault();
                const jobId = btn.dataset.jobId;
                await this.deleteJob(jobId);
            });
        });
    },

    // Show add job modal
    async showAddJobModal() {
        if (!this.jobForm) return;

        // Reset form
        this.jobForm.reset();
        this.currentSkills.clear();
        this.updateSkillTags();
        this.updateFormProgress();
        document.getElementById('jobModalTitle').textContent = 'Add New Job';
        this.jobForm.dataset.jobId = '';
        
        // Reset progress bar
        const progressBar = document.querySelector('.modal-progress-bar');
        if (progressBar) {
            progressBar.style.width = '0%';
        }
        
        // Refresh categories
        await this.loadJobCategories();
        
        // Show modal
        BootstrapInit.showModal('jobModal');
        
        // Focus on first field
        setTimeout(() => {
            const firstField = document.getElementById('jobTitle');
            if (firstField) firstField.focus();
        }, 300);
    },

    // Edit job
    async editJob(jobId) {
        try {
            const data = await APIService.jobs.getById(jobId);
            
            if (!data.success) {
                throw new Error(data.error || 'Failed to load job details');
            }
            
            const job = data.job;

            // Populate form
            document.getElementById('jobTitle').value = job.title;
            document.getElementById('jobDepartment').value = job.department;
            document.getElementById('jobCategory').value = job.category;
            document.getElementById('jobExperience').value = job.experience_level;
            document.getElementById('jobDescription').value = job.description;

            // Update skills
            this.currentSkills = new Set(job.requirements.split(',').map(s => s.trim()).filter(Boolean));
            this.updateSkillTags();
            this.updateFormProgress();

            // Update modal
            document.getElementById('jobModalTitle').textContent = 'Edit Job';
            this.jobForm.dataset.jobId = jobId;
            
            // Show modal
            BootstrapInit.showModal('jobModal');
        } catch (error) {
            console.error('Error loading job details:', error);
            ToastUtils.showError(error.message || 'Failed to load job details');
        }
    },

    // Save job
    async saveJob() {
        // Get form data
        const title = document.getElementById('jobTitle').value.trim();
        const department = document.getElementById('jobDepartment').value.trim();
        const category = document.getElementById('jobCategory').value.trim();
        const experience = document.getElementById('jobExperience').value.trim();
        const description = document.getElementById('jobDescription').value.trim();
        
        // Validate required fields
        const validation = ValidationUtils.validateRequiredFields(
            { title, department, category, experience, description },
            ['title', 'department', 'category', 'experience', 'description']
        );
        
        if (!validation.isValid) {
            ToastUtils.showError('Please fill in all required fields');
            return;
        }
        
        if (this.currentSkills.size === 0) {
            ToastUtils.showError('Please add at least one required skill');
            return;
        }
        
        const jobData = {
            title,
            department,
            category,
            experience_level: experience,
            description,
            requirements: Array.from(this.currentSkills).join(', ')
        };
        
        // Add job ID if editing
        const jobId = this.jobForm.dataset.jobId;
        if (jobId) {
            jobData.id = jobId;
        }
        
        try {
            let result;
            if (jobId) {
                result = await APIService.jobs.update(jobId, jobData);
            } else {
                result = await APIService.jobs.create(jobData);
            }
            
            if (result.success) {
                ToastUtils.showSuccess(`Job ${jobId ? 'updated' : 'created'} successfully`);
                await this.loadJobs();
                BootstrapInit.hideModal('jobModal');
            } else {
                throw new Error(result.error || 'Failed to save job');
            }
        } catch (error) {
            console.error('Error saving job:', error);
            ToastUtils.showError(error.message || 'Failed to save job');
        }
    },

    // Delete job
    async deleteJob(jobId) {
        if (!confirm('Are you sure you want to delete this job?')) return;

        try {
            const result = await APIService.jobs.delete(jobId);
            
            if (result.success) {
                ToastUtils.showSuccess('Job deleted successfully');
                await this.loadJobs();
            } else {
                throw new Error(result.error || 'Failed to delete job');
            }
        } catch (error) {
            console.error('Error deleting job:', error);
            ToastUtils.showError(error.message || 'Failed to delete job');
        }
    },

    // Update skill tags display
    updateSkillTags() {
        if (!this.skillTags) return;

        this.skillTags.innerHTML = Array.from(this.currentSkills).map(skill => `
            <span class="skill-tag">
                ${DOMUtils.escapeHtml(skill)}
                <i class="fas fa-times" data-skill="${DOMUtils.escapeHtml(skill)}"></i>
            </span>
        `).join('');

        // Update skills counter
        this.updateSkillsCounter();

        // Add click handlers for removing skills
        this.skillTags.querySelectorAll('.fa-times').forEach(icon => {
            icon.addEventListener('click', (e) => {
                e.preventDefault();
                const skillTag = icon.closest('.skill-tag');
                skillTag.classList.add('removing');
                
                setTimeout(() => {
                    this.currentSkills.delete(icon.dataset.skill);
                    this.updateSkillTags();
                }, 200);
            });
        });

        // Update form progress
        this.updateFormProgress();
    },

    // Update skills counter
    updateSkillsCounter() {
        const counter = document.getElementById('skillsCounter');
        if (!counter) return;

        const count = this.currentSkills.size;
        counter.textContent = `${count} skill${count !== 1 ? 's' : ''} added`;
        
        if (count === 0) {
            counter.className = 'skills-counter error';
        } else if (count < 3) {
            counter.className = 'skills-counter warning';
        } else {
            counter.className = 'skills-counter';
        }
    },

    // Update form progress
    updateFormProgress() {
        const progressInfo = document.getElementById('formProgress');
        const progressBar = document.querySelector('.modal-progress-bar');
        
        if (!progressInfo || !progressBar) return;

        const title = document.getElementById('jobTitle').value.trim();
        const department = document.getElementById('jobDepartment').value.trim();
        const category = document.getElementById('jobCategory').value.trim();
        const experience = document.getElementById('jobExperience').value.trim();
        const description = document.getElementById('jobDescription').value.trim();
        const skillsCount = this.currentSkills.size;

        const fields = [title, department, category, experience, description].filter(Boolean);
        const progress = ((fields.length + (skillsCount > 0 ? 1 : 0)) / 6) * 100;

        progressBar.style.width = `${progress}%`;

        if (progress === 100) {
            progressInfo.innerHTML = '<i class="fas fa-check-circle text-success me-1"></i>Ready to save';
        } else {
            const missing = [];
            if (!title) missing.push('job title');
            if (!department) missing.push('department');
            if (!category) missing.push('category');
            if (!experience) missing.push('experience level');
            if (skillsCount === 0) missing.push('required skills');
            if (!description) missing.push('description');
            
            progressInfo.innerHTML = `<i class="fas fa-info-circle text-warning me-1"></i>Missing: ${missing.join(', ')}`;
        }
    },

    // Load job categories
    async loadJobCategories() {
        try {
            const data = await APIService.jobCategories.getAll();
            
            if (!data.success) {
                throw new Error(data.error || 'Failed to load job categories');
            }
            
            const categories = data.categories || [];
            const categorySelect = document.getElementById('jobCategory');
            
            if (categorySelect) {
                // Clear existing options except the first one
                while (categorySelect.options.length > 1) {
                    categorySelect.remove(1);
                }
                
                // Add new options
                categories.forEach(category => {
                    const option = document.createElement('option');
                    option.value = category.name;
                    option.textContent = category.name;
                    categorySelect.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Error loading job categories:', error);
            ToastUtils.showError('Failed to load job categories: ' + error.message);
        }
    },

    // Setup category management
    setupCategoryManagement() {
        const addCategoryBtn = document.getElementById('addCategoryBtn');
        const saveCategoryBtn = document.getElementById('saveCategoryBtn');

        if (addCategoryBtn) {
            addCategoryBtn.addEventListener('click', () => {
                document.getElementById('categoryForm').reset();
                BootstrapInit.showModal('categoryModal');
            });
        }

        if (saveCategoryBtn) {
            saveCategoryBtn.addEventListener('click', async () => {
                await this.saveCategory();
            });
        }
    },

    // Save category
    async saveCategory() {
        const nameInput = document.getElementById('categoryName');
        const descInput = document.getElementById('categoryDescription');
        
        if (!nameInput.value.trim()) {
            ToastUtils.showError('Category name is required');
            return;
        }
        
        try {
            const result = await APIService.jobCategories.create({
                name: nameInput.value.trim(),
                description: descInput.value.trim()
            });
            
            if (result.success) {
                ToastUtils.showSuccess('Category created successfully');
                BootstrapInit.hideModal('categoryModal');
                
                // Refresh all category-related UI elements
                await this.loadJobCategories();
                await this.loadJobs();
                
                // Refresh upload section if it exists
                if (typeof loadJobCategoriesForUpload === 'function') {
                    await loadJobCategoriesForUpload();
                }
            } else {
                throw new Error(result.error || 'Failed to create category');
            }
        } catch (error) {
            console.error('Error creating category:', error);
            ToastUtils.showError(error.message || 'Failed to create category');
        }
    }
};

// Make globally available
window.JobsModule = JobsModule;

// Backward compatibility
window.setupJobManagement = JobsModule.init.bind(JobsModule);
