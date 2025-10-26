/**
 * Clean Upload Module
 * Handles file uploads for PDF and XLSX files with batch processing
 * No OCR dependencies - clean and simple implementation
 */

const UploadModule = {
    // Configuration
    config: {
        allowedTypes: ['.pdf', '.xlsx', '.xls'],
        maxFileSize: 16 * 1024 * 1024, // 16MB
        maxFiles: 20,
        uploadEndpoint: '/api/upload-files',
        analysisEndpoint: '/api/start-analysis'
    },

    // State management
    state: {
        selectedJobId: null,
        uploadedFiles: [],
        sessionId: null,
        isUploading: false,
        isAnalyzing: false,
        fileDialogOpened: false
    },

    /**
     * Initialize the upload module
     */
    init() {
        console.log('🚀 Initializing clean upload module');
        
        this.setupEventHandlers();
        this.loadJobPostings();
        
        console.log('✅ Upload module initialized successfully');
    },

    /**
     * Setup event handlers for upload functionality
     */
    setupEventHandlers() {
        const regularUploadZone = document.getElementById('regularUploadZone');
        const bulkUploadZone = document.getElementById('bulkUploadZone');
        const regularFileInput = document.getElementById('regularFileUpload');
        const bulkFileInput = document.getElementById('bulkFileUpload');

        if (regularUploadZone && regularFileInput) {
            // Regular upload zone events
            regularUploadZone.addEventListener('click', (e) => {
                // Don't prevent default - let the click propagate
                console.log('📁 Regular upload zone clicked');
                this.triggerFileUpload('regular');
            });

            regularUploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'copy';
                regularUploadZone.classList.add('drag-over');
            });

            regularUploadZone.addEventListener('dragleave', () => {
                regularUploadZone.classList.remove('drag-over');
            });

            regularUploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                regularUploadZone.classList.remove('drag-over');
                console.log('📂 Files dropped on regular zone');
                this.handleDrop(e, 'regular');
            });

            // File input change event
            regularFileInput.addEventListener('change', (e) => {
                console.log('📄 Regular file input changed');
                this.state.fileDialogOpened = true;
                this.handleFileSelection(e.target.files, 'regular');
            });
            
            // Track focus events to detect dialog opening
            regularFileInput.addEventListener('focus', () => {
                console.log('📄 Regular file input focused - dialog likely opened');
                this.state.fileDialogOpened = true;
            });
            
            regularFileInput.addEventListener('blur', () => {
                console.log('📄 Regular file input blurred - dialog closed');
                setTimeout(() => {
                    this.state.fileDialogOpened = false;
                }, 100);
            });
        }

        if (bulkUploadZone && bulkFileInput) {
            // Bulk upload zone events
            bulkUploadZone.addEventListener('click', (e) => {
                // Don't prevent default - let the click propagate
                console.log('📁 Bulk upload zone clicked');
                this.triggerFileUpload('bulk');
            });

            bulkUploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'copy';
                bulkUploadZone.classList.add('drag-over');
            });

            bulkUploadZone.addEventListener('dragleave', () => {
                bulkUploadZone.classList.remove('drag-over');
            });

            bulkUploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                bulkUploadZone.classList.remove('drag-over');
                console.log('📂 Files dropped on bulk zone');
                this.handleDrop(e, 'bulk');
            });

            // Bulk file input change event
            bulkFileInput.addEventListener('change', (e) => {
                console.log('📄 Bulk file input changed');
                this.state.fileDialogOpened = true;
                this.handleFileSelection(e.target.files, 'bulk');
            });
            
            // Track focus events to detect dialog opening
            bulkFileInput.addEventListener('focus', () => {
                console.log('📄 Bulk file input focused - dialog likely opened');
                this.state.fileDialogOpened = true;
            });
            
            bulkFileInput.addEventListener('blur', () => {
                console.log('📄 Bulk file input blurred - dialog closed');
                setTimeout(() => {
                    this.state.fileDialogOpened = false;
                }, 100);
            });
        }

        // Start analysis button
        const startAnalysisBtn = document.getElementById('startAnalysisBtn');
        if (startAnalysisBtn) {
            startAnalysisBtn.addEventListener('click', () => {
                console.log('🔬 Start analysis clicked');
                this.startAnalysis();
            });
        }

        console.log('✅ Event handlers setup complete');
    },

    /**
     * Trigger file upload dialog
     */
    triggerFileUpload(type) {
        const fileInputId = type === 'regular' ? 'regularFileUpload' : 'bulkFileUpload';
        console.log(`📂 Looking for file input: ${fileInputId}`);
        const fileInput = document.getElementById(fileInputId);
        
        console.log(`📁 File input found:`, !!fileInput);
        if (fileInput) {
            console.log(`📁 File input details:`, {
                id: fileInput.id,
                type: fileInput.type,
                accept: fileInput.accept,
                multiple: fileInput.multiple,
                style: fileInput.style.cssText
            });
        }
        
        if (fileInput) {
            console.log(`📂 Triggering ${type} file dialog`);
            try {
                // First try: Direct click
                fileInput.click();
                
                // Alternative method if direct click fails
                setTimeout(() => {
                    if (!this.state.fileDialogOpened) {
                        console.log(`🔄 Trying alternative file dialog trigger...`);
                        
                        // Method 2: Create and trigger a mouse event
                        const event = new MouseEvent('click', {
                            view: window,
                            bubbles: true,
                            cancelable: true
                        });
                        fileInput.dispatchEvent(event);
                        
                        // Method 3: Focus and simulate enter key
                        setTimeout(() => {
                            if (!this.state.fileDialogOpened) {
                                console.log(`� Trying focus and enter method...`);
                                fileInput.focus();
                                const enterEvent = new KeyboardEvent('keydown', {
                                    key: 'Enter',
                                    code: 'Enter',
                                    keyCode: 13
                                });
                                fileInput.dispatchEvent(enterEvent);
                            }
                        }, 50);
                    }
                }, 100);
                
            } catch (error) {
                console.error(`❌ Error clicking file input:`, error);
            }
        } else {
            console.error(`❌ File input not found for type: ${type}`);
            console.log(`🔍 Available elements with 'Upload' in ID:`, 
                Array.from(document.querySelectorAll('[id*="Upload"]')).map(el => el.id)
            );
        }
    },

    /**
     * Handle drag and drop
     */
    handleDrop(e, type) {
        const files = e.dataTransfer.files;
        console.log(`📂 Handling drop for ${type}: ${files.length} files`);
        this.handleFileSelection(files, type);
    },

    /**
     * Handle file selection
     */
    handleFileSelection(files, type) {
        if (!files || files.length === 0) {
            console.log('ℹ️ No files selected');
            return;
        }

        if (!this.state.selectedJobId) {
            this.showMessage('Please select a job position first', 'warning');
            return;
        }

        console.log(`📄 Processing ${files.length} selected files for ${type} upload`);

        // Validate files
        const validFiles = [];
        const errors = [];

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const validation = this.validateFile(file);
            
            if (validation.isValid) {
                validFiles.push(file);
            } else {
                errors.push(`${file.name}: ${validation.error}`);
            }
        }

        if (validFiles.length === 0) {
            this.showMessage('No valid files to upload', 'error');
            if (errors.length > 0) {
                this.showErrors(errors);
            }
            return;
        }

        if (errors.length > 0) {
            this.showErrors(errors);
        }

        // Upload valid files
        this.uploadFiles(validFiles);
    },

    /**
     * Validate individual file
     */
    validateFile(file) {
        if (!file.name) {
            return { isValid: false, error: 'Invalid file name' };
        }

        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        if (!this.config.allowedTypes.includes(fileExtension)) {
            return { 
                isValid: false, 
                error: `File type not supported. Only ${this.config.allowedTypes.join(', ')} files allowed` 
            };
        }

        if (file.size > this.config.maxFileSize) {
            return { 
                isValid: false, 
                error: `File too large. Maximum size is ${this.config.maxFileSize / (1024*1024)}MB` 
            };
        }

        return { isValid: true };
    },

    /**
     * Upload files to server
     */
    async uploadFiles(files) {
        if (this.state.isUploading) {
            console.log('⏳ Upload already in progress');
            return;
        }

        this.state.isUploading = true;
        this.updateUIState();

        try {
            console.log(`📤 Uploading ${files.length} files to server`);

            const formData = new FormData();
            for (let i = 0; i < files.length; i++) {
                formData.append('files[]', files[i]);
            }
            formData.append('jobId', this.state.selectedJobId);

            const response = await fetch(this.config.uploadEndpoint, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.state.sessionId = result.session_id;
                this.state.uploadedFiles = result.files;
                
                this.showMessage(result.message, 'success');
                this.displayUploadedFiles();
                this.showAnalysisControls();
                
                console.log('✅ Upload successful:', result);
            } else {
                throw new Error(result.error || 'Upload failed');
            }

        } catch (error) {
            console.error('❌ Upload error:', error);
            this.showMessage(`Upload failed: ${error.message}`, 'error');
        } finally {
            this.state.isUploading = false;
            this.updateUIState();
        }
    },

    /**
     * Start analysis of uploaded files
     */
    async startAnalysis() {
        if (!this.state.sessionId) {
            this.showMessage('No files to analyze', 'warning');
            return;
        }

        if (this.state.isAnalyzing) {
            console.log('⏳ Analysis already in progress');
            return;
        }

        this.state.isAnalyzing = true;
        this.updateUIState();

        try {
            console.log('🔬 Starting analysis for session:', this.state.sessionId);

            const response = await fetch(this.config.analysisEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.state.sessionId
                })
            });

            const result = await response.json();

            if (result.success) {
                this.showMessage(result.message, 'success');
                this.displayAnalysisResults(result.results);
                this.resetUploadState();
                
                console.log('✅ Analysis completed:', result);
            } else {
                throw new Error(result.error || 'Analysis failed');
            }

        } catch (error) {
            console.error('❌ Analysis error:', error);
            this.showMessage(`Analysis failed: ${error.message}`, 'error');
        } finally {
            this.state.isAnalyzing = false;
            this.updateUIState();
        }
    },

    /**
     * Load job postings
     */
    async loadJobPostings() {
        try {
            console.log('📋 Loading job postings');
            
            const response = await fetch('/api/job-postings');
            const result = await response.json();

            if (result.success && result.jobs) {
                this.displayJobPostings(result.jobs);
                console.log('✅ Job postings loaded');
            } else {
                console.warn('⚠️ No job postings found');
            }

        } catch (error) {
            console.error('❌ Error loading job postings:', error);
        }
    },

    /**
     * Display job postings
     */
    displayJobPostings(jobs) {
        const container = document.getElementById('jobPostingsContainer');
        if (!container) return;

        container.innerHTML = jobs.map(job => `
            <div class="job-card" onclick="selectJobForUpload(${job.id}, ${JSON.stringify(job)})">
                <h5>${job.position_title}</h5>
                <p class="text-muted">${job.campus_location}</p>
                <p class="small">${job.description ? job.description.substring(0, 100) + '...' : 'No description available'}</p>
            </div>
        `).join('');
    },

    /**
     * Set selected job (called by existing selectJobForUpload function)
     */
    setSelectedJob(jobId) {
        this.state.selectedJobId = jobId;
        console.log('🎯 Job set for upload module:', jobId);
        this.showMessage('Job selected. You can now upload files.', 'info');
    },

    /**
     * Display uploaded files
     */
    displayUploadedFiles() {
        const container = document.getElementById('uploadPreviewContainer');
        if (!container || !this.state.uploadedFiles) return;

        container.innerHTML = this.state.uploadedFiles.map(file => `
            <div class="file-preview-card">
                <i class="${file.icon} ${file.color} me-2"></i>
                <span class="file-name">${file.name}</span>
                <span class="file-type badge bg-secondary ms-2">${file.type}</span>
                <span class="file-status badge bg-success ms-2">${file.status}</span>
            </div>
        `).join('');

        container.style.display = 'block';
    },

    /**
     * Show analysis controls
     */
    showAnalysisControls() {
        const controls = document.getElementById('analysisControls');
        if (controls) {
            controls.style.display = 'block';
        }
    },

    /**
     * Display analysis results
     */
    displayAnalysisResults(results) {
        const container = document.getElementById('analysisResults');
        if (!container || !results) return;

        container.innerHTML = `
            <div class="analysis-summary">
                <h4>Analysis Complete</h4>
                <p>${results.length} candidates processed successfully</p>
            </div>
            <div class="candidates-list">
                ${results.map(result => `
                    <div class="candidate-result-card">
                        <h6>${result.name || result.filename}</h6>
                        <p>Email: ${result.email || 'Not provided'}</p>
                        <p>Match Score: <span class="badge bg-primary">${result.percentage_score}%</span></p>
                    </div>
                `).join('')}
            </div>
        `;

        container.style.display = 'block';
        
        // Redirect to candidates page after a delay
        setTimeout(() => {
            window.location.hash = '#candidates';
        }, 3000);
    },

    /**
     * Reset upload state
     */
    resetUploadState() {
        this.state.sessionId = null;
        this.state.uploadedFiles = [];
        
        const containers = ['uploadPreviewContainer', 'analysisControls'];
        containers.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.style.display = 'none';
        });
    },

    /**
     * Update UI state based on current operations
     */
    updateUIState() {
        const uploadZones = document.querySelectorAll('.upload-zone');
        const startBtn = document.getElementById('startAnalysisBtn');

        uploadZones.forEach(zone => {
            if (this.state.isUploading) {
                zone.classList.add('uploading');
            } else {
                zone.classList.remove('uploading');
            }
        });

        if (startBtn) {
            startBtn.disabled = this.state.isAnalyzing;
            startBtn.innerHTML = this.state.isAnalyzing ? 
                '<i class="fas fa-spinner fa-spin"></i> Analyzing...' : 
                '<i class="fas fa-play"></i> Start Analysis';
        }
    },

    /**
     * Show message to user
     */
    showMessage(message, type = 'info') {
        // You can enhance this with a proper notification system
        const alertClass = {
            success: 'alert-success',
            error: 'alert-danger', 
            warning: 'alert-warning',
            info: 'alert-info'
        }[type] || 'alert-info';

        console.log(`📢 ${type.toUpperCase()}: ${message}`);
        
        // Simple alert for now - can be replaced with toast notifications
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert ${alertClass} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        const container = document.querySelector('.upload-section') || document.body;
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    },

    /**
     * Show validation errors
     */
    showErrors(errors) {
        const errorMsg = 'File validation errors:\n' + errors.join('\n');
        this.showMessage(errorMsg, 'warning');
    }
};

// Make UploadModule globally available
window.UploadModule = UploadModule;

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => UploadModule.init());
} else {
    UploadModule.init();
}