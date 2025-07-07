// Upload Module
const UploadModule = {
    selectedJobId: null,
    selectedFiles: [],

    // Initialize upload functionality
    init() {
        this.setupElements();
        this.setupEventListeners();
        this.loadJobCategories(); // Load job categories on init
    },

    // Setup DOM elements
    setupElements() {
        this.uploadZone = document.getElementById('uploadZone');
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('resumeUpload');
        this.uploadPreview = document.getElementById('uploadPreview');
        this.startUploadBtn = document.getElementById('startUploadBtn');
        this.clearFilesBtn = document.getElementById('clearFilesBtn');
        
        // Debug logging
        console.log('Upload elements found:', {
            uploadZone: !!this.uploadZone,
            uploadArea: !!this.uploadArea,
            fileInput: !!this.fileInput
        });
    },

    // Setup event listeners
    setupEventListeners() {
        if (!this.fileInput) {
            console.error('File input element not found!');
            return;
        }

        // Click to upload - try multiple elements
        const clickableElements = [
            this.uploadZone,
            this.uploadArea,
            document.querySelector('.upload-drop-zone'),
            document.querySelector('[data-upload-zone]')
        ].filter(Boolean);

        clickableElements.forEach(element => {
            if (element) {
                element.addEventListener('click', (e) => {
                    e.preventDefault();
                    console.log('Upload area clicked, opening file browser...');
                    this.fileInput.click();
                });
            }
        });

        // Also try to find and attach to any "click to browse" text
        const browseLinks = document.querySelectorAll('a[href="#"], .browse-link, .click-to-browse');
        browseLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                console.log('Browse link clicked, opening file browser...');
                this.fileInput.click();
            });
        });

        // Global fallback - listen for any click on elements containing "browse" text
        document.addEventListener('click', (e) => {
            const target = e.target;
            const text = target.textContent || '';
            if (text.toLowerCase().includes('browse') || text.toLowerCase().includes('click to')) {
                const uploadSection = target.closest('#upload, .upload-section, [data-section="upload"]');
                if (uploadSection && this.fileInput) {
                    e.preventDefault();
                    console.log('Global browse handler triggered');
                    this.fileInput.click();
                }
            }
        });

        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('drag-over');
        });

        this.uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('drag-over');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('drag-over');
            this.handleFiles(e.dataTransfer.files);
        });

        // File input change
        this.fileInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });

        // Button listeners
        if (this.clearFilesBtn) {
            this.clearFilesBtn.addEventListener('click', () => {
                this.clearSelectedFiles();
            });
        }

        if (this.startUploadBtn) {
            this.startUploadBtn.addEventListener('click', () => {
                this.startUpload();
            });
        }
    },

    // Handle file selection
    handleFiles(files) {
        this.clearSelectedFiles();
        
        const newFiles = Array.from(files).filter(file => {
            const validation = ValidationUtils.validateFile(file);
            
            if (!validation.isValid) {
                validation.errors.forEach(error => {
                    ToastUtils.showWarning(error);
                });
                return false;
            }
            
            return true;
        });

        this.selectedFiles = newFiles;
        this.updatePreview();
    },

    // Update file preview
    updatePreview() {
        if (!this.uploadPreview) return;

        if (this.selectedFiles.length === 0) {
            this.uploadPreview.innerHTML = '';
            this.updateUploadButtonState();
            return;
        }

        this.uploadPreview.innerHTML = this.selectedFiles.map((file, index) => {
            const fileExt = file.name.split('.').pop().toLowerCase();
            const fileIcon = this.getFileIcon(fileExt);
            
            return `
                <div class="file-item" data-index="${index}">
                    <div class="file-icon ${fileExt}">
                        <i class="fas ${fileIcon}"></i>
                    </div>
                    <div class="file-info">
                        <div class="file-name">${DOMUtils.escapeHtml(file.name)}</div>
                        <div class="file-details">
                            <span class="file-size">${FormatUtils.formatFileSize(file.size)}</span>
                            <span class="file-type">${fileExt.toUpperCase()}</span>
                        </div>
                    </div>
                    <div class="file-actions">
                        <button class="remove-file" data-index="${index}" title="Remove file">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </div>
            `;
        }).join('');

        this.updateUploadButtonState();
        this.setupRemoveFileListeners();
        
        // Show reminder to select job if none is selected
        if (!this.selectedJobId && this.selectedFiles.length > 0) {
            ToastUtils.showInfo('Files ready! Please select a job position above to continue.');
        }
    },

    // Setup remove file listeners
    setupRemoveFileListeners() {
        document.querySelectorAll('.remove-file').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const index = parseInt(btn.dataset.index);
                this.selectedFiles.splice(index, 1);
                this.updatePreview();
            });
        });
    },

    // Get file icon class
    getFileIcon(extension) {
        switch (extension) {
            case 'pdf': return 'fa-file-pdf';
            case 'docx':
            case 'doc': return 'fa-file-word';
            case 'txt': return 'fa-file-alt';
            default: return 'fa-file';
        }
    },

    // Start upload process
    async startUpload() {
        if (this.selectedFiles.length === 0) {
            ToastUtils.showWarning('Please select files to upload');
            return;
        }

        if (!this.selectedJobId) {
            ToastUtils.showWarning('Please select a job position first');
            // Scroll to job selection area if it exists
            const jobCategoriesSection = document.getElementById('jobCategoriesUpload');
            if (jobCategoriesSection) {
                jobCategoriesSection.scrollIntoView({ behavior: 'smooth' });
            }
            return;
        }

        this.showLoadingState(true);

        try {
            const result = await APIService.uploadFiles([...this.selectedFiles], this.selectedJobId);

            if (result.success) {
                let message = `Successfully processed ${result.results.length} resumes`;
                if (result.warnings && result.warnings.length > 0) {
                    message += ` (${result.warnings.length} files had issues)`;
                }
                
                ToastUtils.showToast(message, result.warnings ? 'warning' : 'success');
                this.displayRankingResults(result.results);
                this.updateResultsSummary(result.results);
                
                this.clearSelectedFiles();
                this.fileInput.value = '';
            } else {
                throw new Error(result.error || 'Upload failed');
            }

        } catch (error) {
            console.error('Upload error:', error);
            ToastUtils.showError(`Upload failed: ${error.message}`);
        } finally {
            this.showLoadingState(false);
        }
    },

    // Show/hide loading state
    showLoadingState(loading) {
        if (!this.startUploadBtn) return;
        
        const btnText = this.startUploadBtn.querySelector('span');
        const btnLoader = this.startUploadBtn.querySelector('.btn-loader');
        
        this.startUploadBtn.disabled = loading;
        if (btnText) btnText.style.display = loading ? 'none' : 'inline';
        if (btnLoader) btnLoader.style.display = loading ? 'inline-block' : 'none';
    },

    // Select job for upload
    selectJob(jobId) {
        const job = window.jobsData && window.jobsData[jobId];
        if (!job) {
            ToastUtils.showError('Job not found');
            return;
        }
        
        this.selectedJobId = jobId;
        this.updateSelectedJobUI(job);
        ToastUtils.showSuccess(`Selected job: ${job.title}`);
    },

    // Update selected job UI
    updateSelectedJobUI(job) {
        const detailsSection = document.getElementById('selectedJobDetails');
        if (detailsSection) {
            detailsSection.style.display = 'block';
            detailsSection.querySelector('.job-title').innerHTML = `<strong><i class="fas fa-briefcase me-2"></i>Position:</strong> ${DOMUtils.escapeHtml(job.title)}`;
            detailsSection.querySelector('.job-description').innerHTML = `<strong><i class="fas fa-info-circle me-2"></i>Description:</strong> ${DOMUtils.escapeHtml(job.description)}`;
            detailsSection.querySelector('.required-skills').innerHTML = `<strong><i class="fas fa-tools me-2"></i>Required Skills:</strong> ${DOMUtils.escapeHtml(job.requirements)}`;
        }
        
        // Update card selection states
        document.querySelectorAll('.job-category-card').forEach(card => {
            card.classList.remove('selected');
        });
        
        const selectedCard = document.querySelector(`[data-job-id="${this.selectedJobId}"]`);
        if (selectedCard) {
            selectedCard.classList.add('selected');
        }
        
        // Show upload zone
        const uploadInstructions = document.getElementById('uploadInstructions');
        const uploadZone = document.getElementById('uploadZone');
        if (uploadInstructions) uploadInstructions.style.display = 'none';
        if (uploadZone) uploadZone.style.display = 'block';
    },

    // Clear selected files
    clearSelectedFiles() {
        this.selectedFiles = [];
        if (this.fileInput) this.fileInput.value = '';
        if (this.uploadPreview) this.uploadPreview.innerHTML = '';
        this.updateUploadButtonState();
    },

    // Update upload button state
    updateUploadButtonState() {
        const uploadActions = document.getElementById('uploadActions');
        
        if (this.selectedFiles.length > 0 && this.selectedJobId) {
            if (this.startUploadBtn) this.startUploadBtn.disabled = false;
            if (uploadActions) uploadActions.style.display = 'block';
            this.updateFileStats();
        } else {
            if (this.startUploadBtn) this.startUploadBtn.disabled = true;
            if (this.selectedFiles.length === 0 && uploadActions) {
                uploadActions.style.display = 'none';
            }
        }
    },

    // Update file statistics
    updateFileStats() {
        const fileCount = document.getElementById('fileCount');
        const totalSize = document.getElementById('totalSize');
        
        if (fileCount) {
            fileCount.textContent = `${this.selectedFiles.length} file${this.selectedFiles.length !== 1 ? 's' : ''} selected`;
        }
        
        if (totalSize) {
            const total = this.selectedFiles.reduce((sum, file) => sum + file.size, 0);
            totalSize.textContent = `${(total / (1024 * 1024)).toFixed(2)} MB`;
        }
    },

    // Display ranking results
    displayRankingResults(results) {
        const resultsCard = document.getElementById('resultsCard');
        const resultsList = document.getElementById('rankingResults');
        
        if (!resultsCard || !resultsList) return;
        
        // Sort results by match score
        results.sort((a, b) => b.matchScore - a.matchScore);
        
        resultsList.innerHTML = results.map(result => `
            <div class="ranking-result-card ${DOMUtils.getScoreColorClass(result.matchScore)}">
                <div class="result-header">
                    <div class="result-title">
                        <i class="fas fa-file-alt me-2"></i>
                        <h4>${DOMUtils.escapeHtml(result.filename)}</h4>
                    </div>
                    <div class="match-score">
                        <div class="score-circle">
                            ${result.matchScore}%
                        </div>
                    </div>
                </div>
                <div class="result-body">
                    <div class="candidate-info">
                        ${result.name ? `<p><strong>Name:</strong> ${DOMUtils.escapeHtml(result.name)}</p>` : ''}
                        ${result.email ? `<p><strong>Email:</strong> ${DOMUtils.escapeHtml(result.email)}</p>` : ''}
                        ${result.phone ? `<p><strong>Phone:</strong> ${DOMUtils.escapeHtml(result.phone)}</p>` : ''}
                        ${result.predictedCategory ? `
                            <p><strong>Predicted Category:</strong> 
                                <span class="category-prediction">
                                    ${DOMUtils.escapeHtml(result.predictedCategory.category)} 
                                    <span class="confidence-score">(${result.predictedCategory.confidence}% confidence)</span>
                                </span>
                            </p>
                        ` : ''}
                    </div>
                    <div class="skills-section">
                        <h5>Matched Skills</h5>
                        <div class="skills-list">
                            ${result.matchedSkills.map(skill => 
                                `<span class="skill-tag matched">${DOMUtils.escapeHtml(skill)}</span>`
                            ).join('')}
                        </div>
                    </div>
                    <div class="missing-skills-section">
                        <h5>Missing Skills</h5>
                        <div class="skills-list">
                            ${result.missingSkills.map(skill => 
                                `<span class="skill-tag missing">${DOMUtils.escapeHtml(skill)}</span>`
                            ).join('')}
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
        
        resultsCard.style.display = 'block';
    },

    // Update results summary (placeholder)
    updateResultsSummary(results) {
        // Implementation would go here - keeping existing functionality  
        if (typeof updateResultsSummary === 'function') {
            updateResultsSummary(results);
        }
    },

    // Load job categories for upload selection
    async loadJobCategories() {
        try {
            const response = await fetch('/api/jobs');
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || 'Failed to load jobs');
            }
            
            const jobs = data.jobs || [];
            const categoriesList = document.getElementById('jobCategoriesUpload');
            
            if (!categoriesList) {
                console.error('Element jobCategoriesUpload not found!');
                return;
            }
            
            if (jobs.length === 0) {
                categoriesList.innerHTML = `
                    <div class="no-jobs-message">
                        <div class="no-jobs-icon">
                            <i class="fas fa-briefcase"></i>
                        </div>
                        <h4>No Jobs Available</h4>
                        <p>Please go to the "Job Requirements" section and create some job positions first.</p>
                        <a href="#jobs" class="btn btn-primary" onclick="window.showSection('jobs')">
                            <i class="fas fa-plus me-2"></i>Create Jobs
                        </a>
                    </div>
                `;
                return;
            }
            
            categoriesList.innerHTML = jobs.map(job => `
                <div class="job-category-card" data-job-id="${job.id}" onclick="selectJobForUpload(${job.id})">
                    <div class="job-category-header">
                        <h4>${DOMUtils.escapeHtml(job.title)}</h4>
                        <span class="badge">${DOMUtils.escapeHtml(job.category)}</span>
                    </div>
                    <div class="job-category-body">
                        <p class="job-category-description">${DOMUtils.escapeHtml(job.description.substring(0, 150))}${job.description.length > 150 ? '...' : ''}</p>
                        <div class="job-category-skills">
                            ${job.requirements.split(',').map(skill => 
                                `<span class="skill-tag">${DOMUtils.escapeHtml(skill.trim())}</span>`
                            ).join('')}
                        </div>
                    </div>
                    <div class="job-category-footer">
                        <button class="btn btn-primary select-job" onclick="event.stopPropagation(); selectJobForUpload(${job.id})">
                            <i class="fas fa-check me-2"></i>Select Job
                        </button>
                    </div>
                </div>
            `).join('');
            
            // Store jobs data for later use
            window.jobsData = jobs.reduce((acc, job) => {
                acc[job.id] = job;
                return acc;
            }, {});
            
        } catch (error) {
            console.error('Error loading jobs for upload:', error);
            const categoriesList = document.getElementById('jobCategoriesUpload');
            if (categoriesList) {
                categoriesList.innerHTML = `
                    <div class="error-message">
                        <div class="error-icon">
                            <i class="fas fa-exclamation-triangle"></i>
                        </div>
                        <h4>Failed to Load Jobs</h4>
                        <p>Error: ${error.message}</p>
                        <button class="btn btn-outline-primary" onclick="UploadModule.loadJobCategories()">
                            <i class="fas fa-redo me-2"></i>Retry
                        </button>
                    </div>
                `;
            }
            ToastUtils.showError('Failed to load jobs: ' + error.message);
        }
    },
};

// Make globally available
window.UploadModule = UploadModule;
window.selectedJobId = null; // For backward compatibility
window.selectedFiles = []; // For backward compatibility

// Backward compatibility functions
window.selectJobForUpload = function(jobId) {
    UploadModule.selectJob(jobId);
    // Update global variables for compatibility
    window.selectedJobId = UploadModule.selectedJobId;
};

window.setupResumeUpload = UploadModule.init.bind(UploadModule);

// Add missing global functions for backward compatibility
window.loadJobCategoriesForUpload = function() {
    UploadModule.loadJobCategories();
};

// Global function to trigger file browser
window.triggerFileUpload = function() {
    console.log('Global triggerFileUpload called');
    const fileInput = document.getElementById('resumeUpload');
    if (fileInput) {
        fileInput.click();
    } else {
        console.error('File input not found!');
    }
};

// Global function for "click to browse" links
window.openFileBrowser = window.triggerFileUpload;
