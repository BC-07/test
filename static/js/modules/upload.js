// Enhanced Upload Module with simplified digital document focus
const UploadModule = {
    selectedJobId: null,
    selectedFiles: [],
    processingType: 'pds',  // Always use PDS processing
    uploadMode: 'regular', // Always use regular (digital documents only)
    isUploading: false,
    isAnalyzing: false,  // Track analysis state
    currentBatch: null,  // Store batch info for analysis
    jobs: [], // Store loaded jobs for reference

    // Initialize upload functionality
    init() {
        console.log('🚀 UploadModule.init() called');
        this.setupElements();
        this.setupEventListeners();
        this.loadJobPostings();
        // Always use PDS processing - no selector needed
        this.processingType = 'pds';
        // Always use regular mode - digital documents only
        this.uploadMode = 'regular';
        console.log('✅ UploadModule initialized successfully');
    },

    // Initialize PDS upload specifically
    initPDSUpload() {
        this.processingType = 'pds';
        this.uploadMode = 'regular';
        this.setupElements();
        this.setupEventListeners();
        this.loadJobPostings();
    },

    // No longer needed - removed OCR tab switching
    // Upload mode is always 'regular' for digital documents
            if (uploadStats) uploadStats.textContent = 'OCR';
        }
        
        // Re-setup elements for the new mode
        this.setupFileInput();
        this.setupUploadZone();
        
        // If a job is already selected, update the display for the new mode
        if (this.selectedJobId) {
            this.updateJobSelectionDisplay();
        }
        
        if (typeof ToastUtils !== 'undefined') {
            const modeText = mode === 'regular' ? 'Regular Files' : 'Scanned Documents (OCR)';
            ToastUtils.showInfo(`Switched to ${modeText} upload mode`);
        }
    },

    // Setup DOM elements and event listeners
    setupElements() {
        this.setupFileInput();
        this.setupUploadZone();
        this.setupButtons();
    },

    setupFileInput() {
        // Setup file input for digital documents only
        const regularInput = document.getElementById('regularFileUpload');
        
        if (regularInput) {
            // Remove any existing listeners to prevent duplicates
            const boundHandler = this.handleFileSelection.bind(this);
            regularInput.removeEventListener('change', boundHandler);
            regularInput.addEventListener('change', boundHandler);
        }
        
        // Legacy support for existing resumeUpload input
        const legacyInput = document.getElementById('resumeUpload');
        
        if (legacyInput && !regularInput) {
            const boundHandler = this.handleFileSelection.bind(this);
            legacyInput.removeEventListener('change', boundHandler);
            legacyInput.addEventListener('change', boundHandler);
        }
    },

    setupUploadZone() {
        // Setup upload zone for digital documents only
        const regularZone = document.getElementById('regularUploadZone');
        
        // Setup regular upload zone
        if (regularZone) {
            this.setupZoneEvents(regularZone, 'regular');
        }
        
        // Also try legacy IDs for backward compatibility
        const legacyZoneIds = ['resumeUploadZone', 'uploadZone'];
        for (const id of legacyZoneIds) {
            const uploadZone = document.getElementById(id);
            if (uploadZone) {
                this.setupZoneEvents(uploadZone, 'regular');
            }
        }
    },

    setupZoneEvents(uploadZone, mode) {
        if (!uploadZone) {
            console.log('❌ setupZoneEvents: uploadZone is null');
            return;
        }
        
        console.log('🎯 Setting up events for upload zone:', uploadZone.id);
        
        // Remove any existing onclick handlers
        uploadZone.removeAttribute('onclick');
        
        // Add proper event listeners with bound context
        uploadZone.addEventListener('click', () => {
            console.log('🖱️ Upload zone clicked!');
            this.triggerFileUpload(mode);
        });
        uploadZone.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadZone.addEventListener('drop', this.handleDrop.bind(this));
        
        // Also setup browse link
        const browseLink = uploadZone.querySelector('.browse-link');
        if (browseLink) {
            console.log('🔗 Setting up browse link');
            browseLink.removeAttribute('onclick');
            browseLink.addEventListener('click', (e) => {
                console.log('🔗 Browse link clicked!');
                e.stopPropagation();
                this.triggerFileUpload(mode);
            });
        }
        
        console.log('✅ Event handlers set up for:', uploadZone.id);
    },

    setupButtons() {
        // Setup clear button
        const clearBtnIds = ['resumeClearFilesBtn', 'clearFilesBtn'];
        for (const id of clearBtnIds) {
            const clearBtn = document.getElementById(id);
            if (clearBtn) {
                clearBtn.addEventListener('click', () => this.clearFiles());
                break;
            }
        }

        // Setup upload button
        const uploadBtnIds = ['resumeStartUploadBtn', 'startUploadBtn'];
        for (const id of uploadBtnIds) {
            const uploadBtn = document.getElementById(id);
            if (uploadBtn) {
                uploadBtn.addEventListener('click', () => this.startUpload());
                break;
            }
        }
    },

    setupEventListeners() {
        // This method is called from init and initResumeUpload
        // Most event listeners are now set up in setupElements
    },

    // Trigger file upload dialog
    triggerFileUpload(mode = 'regular') {
        console.log('🚀 triggerFileUpload called with mode:', mode);
        
        // Always use regular mode for digital documents
        let fileInput = document.getElementById('regularFileUpload');
        console.log('📍 Found regularFileUpload:', fileInput);
        
        // Fallback to legacy IDs if new ones not found
        if (!fileInput) {
            console.log('⚠️ Trying fallback file input IDs...');
            const fileInputIds = ['resumeUpload', 'resumeFileUpload', 'fileUpload'];
            for (const id of fileInputIds) {
                fileInput = document.getElementById(id);
                if (fileInput) {
                    console.log(`✅ Found fallback: ${id}`);
                    break;
                }
            }
        }
        
        if (fileInput) {
            console.log('✅ Triggering file input click');
            fileInput.click();
            console.log('✅ File input click completed');
        } else {
            console.error('❌ No file input found!');
        }
    },

    // Handle file selection from input
    handleFileSelection(event) {
        const files = Array.from(event.target.files);
        if (files.length === 0) return;
        
        // Clear existing files when new files are selected
        this.selectedFiles = [];
        this.addFiles(files);
        
        // Clear the input to allow selecting the same files again
        event.target.value = '';
    },

    // Handle drag over
    handleDragOver(event) {
        event.preventDefault();
        event.stopPropagation();
        event.dataTransfer.dropEffect = 'copy';
        
        const uploadZone = event.currentTarget;
        uploadZone.classList.add('drag-over');
    },

    // Handle drag leave
    handleDragLeave(event) {
        event.preventDefault();
        event.stopPropagation();
        
        const uploadZone = event.currentTarget;
        uploadZone.classList.remove('drag-over');
    },

    // Handle file drop
    handleDrop(event) {
        event.preventDefault();
        event.stopPropagation();
        
        const uploadZone = event.currentTarget;
        uploadZone.classList.remove('drag-over');
        
        const files = Array.from(event.dataTransfer.files);
        
        // Clear existing files when new files are dropped, same as file selection
        this.selectedFiles = [];
        this.addFiles(files);
    },

    // Add files to the selection
    addFiles(newFiles) {
        if (!Array.isArray(newFiles)) return;

        // Filter files based on upload mode
        let allowedExtensions;
        let expectedFormats;
        
        if (this.uploadMode === 'ocr') {
            allowedExtensions = ['jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp', 'gif', 'pdf'];
            expectedFormats = 'JPG, PNG, TIFF, BMP, PDF files for OCR processing';
        } else {
            // Always use PDS processing - support Excel PDS and document formats
            allowedExtensions = ['xlsx', 'xls', 'pdf', 'docx', 'txt'];
            expectedFormats = 'Excel (.xlsx, .xls), PDF, DOCX, or TXT files';
        }

        const validFiles = newFiles.filter(file => {
            const extension = file.name.toLowerCase().split('.').pop();
            return allowedExtensions.includes(extension);
        });

        if (validFiles.length !== newFiles.length) {
            if (typeof ToastUtils !== 'undefined') {
                ToastUtils.showWarning(`Only ${expectedFormats} are supported`);
            }
        }

        // Add valid files
        this.selectedFiles.push(...validFiles);
        this.updateFilePreview();
        this.updateUploadButton();
    },

    // Update file preview display
    updateFilePreview() {
        // Select preview container based on current upload mode
        let preview = null;
        if (this.uploadMode === 'ocr') {
            preview = document.getElementById('ocrUploadPreview');
        } else {
            preview = document.getElementById('regularUploadPreview');
        }
        
        // Fallback to legacy IDs if new ones not found
        if (!preview) {
            const previewIds = ['resumeUploadPreview', 'uploadPreview'];
            for (const id of previewIds) {
                preview = document.getElementById(id);
                if (preview) break;
            }
        }

        if (!preview) return;

        if (this.selectedFiles.length === 0) {
            preview.style.display = 'none';
            this.hideUploadActions();
            return;
        }

        // Show file list
        preview.innerHTML = this.selectedFiles.map((file, index) => `
            <div class="file-item" data-index="${index}">
                <div class="file-icon">
                    <i class="fas ${this.getFileIcon(file.name)}"></i>
                </div>
                <div class="file-info">
                    <span class="file-name">${file.name}</span>
                    <span class="file-size">${this.formatFileSize(file.size)}</span>
                </div>
                <button class="btn btn-sm btn-outline-danger" onclick="UploadModule.removeFile(${index})">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `).join('');

        preview.style.display = 'block';
        this.showUploadActions();
        this.updateUploadStats();
    },

    // Get appropriate icon for file type
    getFileIcon(filename) {
        const extension = filename.toLowerCase().split('.').pop();
        switch (extension) {
            case 'pdf': return 'fa-file-pdf text-danger';
            case 'docx': case 'doc': return 'fa-file-word text-primary';
            case 'xlsx': case 'xls': return 'fa-file-excel text-success';
            case 'txt': return 'fa-file-alt text-secondary';
            case 'jpg': case 'jpeg': case 'png': case 'gif': case 'bmp': case 'tiff': case 'tif': 
                return 'fa-file-image text-warning';
            default: return 'fa-file text-muted';
        }
    },

    // Format file size for display
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    // Remove file from selection
    removeFile(index) {
        this.selectedFiles.splice(index, 1);
        this.updateFilePreview();
        this.updateUploadButton();
    },

    // Clear all files
    clearFiles() {
        this.selectedFiles = [];
        this.updateFilePreview();
        this.updateUploadButton();

        // Reset all file inputs
        const fileInputIds = ['regularFileUpload', 'ocrFileUpload', 'resumeUpload', 'resumeFileUpload', 'fileUpload'];
        for (const id of fileInputIds) {
            const fileInput = document.getElementById(id);
            if (fileInput) {
                fileInput.value = '';
            }
        }
    },

    // Reset entire upload state (new method)
    resetUploadState() {
        this.clearFiles();
        this.currentBatch = null;
        this.isUploading = false;
        this.isAnalyzing = false;
        
        // Reset upload button visibility and state
        const uploadBtnIds = ['resumeStartUploadBtn', 'startUploadBtn'];
        for (const id of uploadBtnIds) {
            const uploadBtn = document.getElementById(id);
            if (uploadBtn) {
                const loader = uploadBtn.querySelector('.btn-loader');
                const span = uploadBtn.querySelector('span');
                const icon = uploadBtn.querySelector('i:not(.btn-loader i)');
                
                if (loader) loader.style.display = 'none';
                if (span) span.textContent = 'Upload Files';
                if (icon) {
                    icon.className = 'fas fa-upload me-2';
                }
                
                uploadBtn.style.display = 'block';
                uploadBtn.disabled = true; // Will be enabled when files are selected
                uploadBtn.className = 'btn btn-primary btn-lg'; // Reset to primary color
                
                // Remove analysis listener and add upload listener
                const newBtn = uploadBtn.cloneNode(true);
                uploadBtn.parentNode.replaceChild(newBtn, uploadBtn);
                newBtn.addEventListener('click', () => this.startUpload());
                
                break;
            }
        }
        
        // Hide results step
        const resultStepIds = ['resumeResultsStep', 'resultsStep'];
        for (const id of resultStepIds) {
            const resultsStep = document.getElementById(id);
            if (resultsStep) {
                resultsStep.style.display = 'none';
                break;
            }
        }
    },

    // Show upload actions
    showUploadActions() {
        const actionIds = ['resumeUploadActions', 'uploadActions'];
        for (const id of actionIds) {
            const actions = document.getElementById(id);
            if (actions) {
                actions.style.display = 'block';
                break;
            }
        }
    },

    // Hide upload actions
    hideUploadActions() {
        const actionIds = ['resumeUploadActions', 'uploadActions'];
        for (const id of actionIds) {
            const actions = document.getElementById(id);
            if (actions) {
                actions.style.display = 'none';
                break;
            }
        }
    },

    // Update upload statistics
    updateUploadStats() {
        const fileCountIds = ['resumeFileCount', 'fileCount'];
        const totalSizeIds = ['resumeTotalSize', 'totalSize'];
        
        // Update file count
        for (const id of fileCountIds) {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = `${this.selectedFiles.length} file${this.selectedFiles.length !== 1 ? 's' : ''} selected`;
                break;
            }
        }

        // Update total size
        const totalBytes = this.selectedFiles.reduce((sum, file) => sum + file.size, 0);
        for (const id of totalSizeIds) {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = this.formatFileSize(totalBytes);
                break;
            }
        }
    },

    // Update upload button state and text
    updateUploadButton() {
        const uploadBtnIds = ['resumeStartUploadBtn', 'startUploadBtn'];
        
        for (const id of uploadBtnIds) {
            const uploadBtn = document.getElementById(id);
            if (uploadBtn) {
                const shouldEnable = this.selectedFiles.length > 0 && this.selectedJobId;
                uploadBtn.disabled = !shouldEnable;
                
                // Update button text to reflect direct analysis
                const span = uploadBtn.querySelector('span');
                const icon = uploadBtn.querySelector('i:not(.btn-loader i)');
                
                if (shouldEnable) {
                    if (span) span.textContent = `Start Analysis (${this.selectedFiles.length} file${this.selectedFiles.length !== 1 ? 's' : ''})`;
                    if (icon) icon.className = 'fas fa-chart-line me-2';
                    uploadBtn.classList.remove('btn-secondary');
                    uploadBtn.classList.add('btn-primary');
                } else if (!this.selectedJobId) {
                    if (span) span.textContent = 'Select a job position first';
                    if (icon) icon.className = 'fas fa-briefcase me-2';
                    uploadBtn.classList.remove('btn-primary');
                    uploadBtn.classList.add('btn-secondary');
                } else {
                    if (span) span.textContent = 'Select files to analyze';
                    if (icon) icon.className = 'fas fa-upload me-2';
                    uploadBtn.classList.remove('btn-primary');
                    uploadBtn.classList.add('btn-secondary');
                }
                break;
            }
        }
    },

    // Start complete upload and analysis process (simplified single-step flow)
    async startUpload() {
        if (this.selectedFiles.length === 0 || !this.selectedJobId || this.isUploading) {
            return;
        }

        this.isUploading = true;
        this.showUploadProgress();

        try {
            const formData = new FormData();
            
            // Add files
            this.selectedFiles.forEach(file => {
                formData.append('files[]', file);
            });
            
            // Add job ID
            formData.append('jobId', this.selectedJobId);
            
            // Determine endpoint based on upload mode
            let endpoint;
            if (this.uploadMode === 'ocr') {
                endpoint = '/api/upload-ocr';
            } else {
                endpoint = '/api/upload-pds'; // Use existing PDS endpoint
            }

            // Show processing message
            this.updateProgressMessage('Processing and analyzing files...');

            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                // Show final results directly (no two-step process needed)
                this.showCompleteResults(result);
                this.clearFiles();
                
                if (typeof ToastUtils !== 'undefined') {
                    ToastUtils.showSuccess(`Successfully analyzed ${result.results.length} files!`);
                }
            } else {
                throw new Error(result.error || 'Processing failed');
            }

        } catch (error) {
            console.error('Processing error:', error);
            if (typeof ToastUtils !== 'undefined') {
                ToastUtils.showError(`Processing failed: ${error.message}`);
            }
        } finally {
            this.isUploading = false;
            this.hideUploadProgress();
        }
    },

    // Start analysis process (Step 2: Assessment)
    async startAnalysis() {
        if (!this.currentBatch || this.isAnalyzing) {
            return;
        }

        this.isAnalyzing = true;
        this.showAnalysisProgress();

        try {
            const response = await fetch('/api/start-analysis', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    batch_id: this.currentBatch.batch_id,
                    job_id: this.currentBatch.job_id
                })
            });

            const result = await response.json();

            if (result.success) {
                this.showAnalysisResults(result);
                
                if (typeof ToastUtils !== 'undefined') {
                    ToastUtils.showSuccess(`Analysis complete! ${result.results.completed} candidates assessed.`);
                }
            } else {
                throw new Error(result.error || 'Analysis failed');
            }

        } catch (error) {
            console.error('Analysis error:', error);
            if (typeof ToastUtils !== 'undefined') {
                ToastUtils.showError(`Analysis failed: ${error.message}`);
            }
        } finally {
            this.isAnalyzing = false;
            this.hideAnalysisProgress();
        }
    },

    // Show upload progress
    showUploadProgress() {
        const resultStepIds = ['resumeResultsStep', 'resultsStep'];
        for (const id of resultStepIds) {
            const resultsStep = document.getElementById(id);
            if (resultsStep) {
                resultsStep.style.display = 'block';
                resultsStep.scrollIntoView({ behavior: 'smooth' });
                break;
            }
        }

        // Show loading state on button
        const uploadBtnIds = ['resumeStartUploadBtn', 'startUploadBtn'];
        for (const id of uploadBtnIds) {
            const uploadBtn = document.getElementById(id);
            if (uploadBtn) {
                const loader = uploadBtn.querySelector('.btn-loader');
                const span = uploadBtn.querySelector('span');
                if (loader) loader.style.display = 'inline-block';
                if (span) span.textContent = 'Processing...';
                uploadBtn.disabled = true;
                break;
            }
        }
    },

    // Hide upload progress
    hideUploadProgress() {
        const uploadBtnIds = ['resumeStartUploadBtn', 'startUploadBtn'];
        for (const id of uploadBtnIds) {
            const uploadBtn = document.getElementById(id);
            if (uploadBtn) {
                const loader = uploadBtn.querySelector('.btn-loader');
                const span = uploadBtn.querySelector('span');
                if (loader) loader.style.display = 'none';
                
                // If we have a current batch (successful upload), hide the upload button
                if (this.currentBatch) {
                    uploadBtn.style.display = 'none';
                } else {
                    // Normal state - reset to upload
                    if (span) span.textContent = 'Upload Files';
                    uploadBtn.disabled = this.selectedFiles.length === 0;
                    uploadBtn.style.display = 'block';
                }
                break;
            }
        }
    },

    // Show analysis progress
    showAnalysisProgress() {
        const uploadBtnIds = ['resumeStartUploadBtn', 'startUploadBtn'];
        for (const id of uploadBtnIds) {
            const uploadBtn = document.getElementById(id);
            if (uploadBtn) {
                const loader = uploadBtn.querySelector('.btn-loader');
                const span = uploadBtn.querySelector('span');
                if (loader) loader.style.display = 'inline-block';
                if (span) span.textContent = 'Analyzing...';
                uploadBtn.disabled = true;
                break;
            }
        }
    },

    // Hide analysis progress
    hideAnalysisProgress() {
        const uploadBtnIds = ['resumeStartUploadBtn', 'startUploadBtn'];
        for (const id of uploadBtnIds) {
            const uploadBtn = document.getElementById(id);
            if (uploadBtn) {
                const loader = uploadBtn.querySelector('.btn-loader');
                const span = uploadBtn.querySelector('span');
                if (loader) loader.style.display = 'none';
                if (span) span.textContent = 'Start Analysis';
                uploadBtn.disabled = false;
                break;
            }
        }
    },

    // Show extraction results (Step 1 complete)
    showExtractionResults(data) {
        console.log('Extraction results:', data);
        
        // Find and show the results step
        const resultStepIds = ['resumeResultsStep', 'resultsStep'];
        let resultsStep = null;
        
        for (const id of resultStepIds) {
            resultsStep = document.getElementById(id);
            if (resultsStep) {
                resultsStep.style.display = 'block';
                break;
            }
        }
        
        if (!resultsStep) {
            console.error('Results step element not found');
            return;
        }

        // Update summary with extraction data
        const processedCount = data.extraction_summary.successful_extractions;
        const totalFiles = data.extraction_summary.total_files;
        const failedFiles = data.extraction_summary.failed_extractions;

        const processedCountEl = document.getElementById('processedCount');
        const avgScoreEl = document.getElementById('avgScore');
        const topCandidatesEl = document.getElementById('topCandidates');
        
        if (processedCountEl) processedCountEl.textContent = processedCount;
        if (avgScoreEl) avgScoreEl.textContent = 'Pending Analysis';
        if (topCandidatesEl) topCandidatesEl.textContent = 'Pending Analysis';

        // Show extraction summary
        this.displayExtractionSummary(data);

        // Show analysis button
        this.showAnalysisButton();
    },

    // Show analysis results (Step 2 complete)
    showAnalysisResults(data) {
        console.log('Analysis results:', data);

        const results = data.results.assessments || [];
        const processedCount = results.length;
        const avgScore = results.length > 0 ? 
            Math.round(results.reduce((sum, r) => sum + (r.score || 0), 0) / results.length) : 0;
        const topCandidates = results.filter(r => (r.score || 0) >= 80).length;

        // Update summary statistics
        const avgScoreEl = document.getElementById('avgScore');
        const topCandidatesEl = document.getElementById('topCandidates');
        
        if (avgScoreEl) avgScoreEl.textContent = avgScore + '%';
        if (topCandidatesEl) topCandidatesEl.textContent = topCandidates;

        // Display analysis results
        this.displayAnalysisResults(data);

        // Hide analysis button, show completion message
        this.hideAnalysisButton();
        this.showCompletionMessage();
    },

    // Display extraction summary
    displayExtractionSummary(data) {
        const resultsListEl = document.getElementById('rankingResults');
        if (resultsListEl) {
            const summary = data.extraction_summary;
            resultsListEl.innerHTML = `
                <div class="extraction-summary">
                    <h4>📄 File Processing Summary</h4>
                    <div class="summary-stats">
                        <div class="stat-item">
                            <span class="stat-label">Files Uploaded:</span>
                            <span class="stat-value">${summary.total_files}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Successfully Extracted:</span>
                            <span class="stat-value">${summary.successful_extractions}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Failed Extractions:</span>
                            <span class="stat-value">${summary.failed_extractions}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Target Position:</span>
                            <span class="stat-value">${data.job_info.title}</span>
                        </div>
                    </div>
                    
                    <div class="files-processed">
                        <h5>📋 Processed Files:</h5>
                        ${summary.files_processed.map(file => `
                            <div class="file-item ${file.extraction_successful ? 'success' : 'error'}">
                                <span class="file-name">${file.filename}</span>
                                <span class="file-status">${file.extraction_successful ? '✅ Extracted' : '❌ Failed'}</span>
                                ${file.candidate_name ? `<span class="candidate-name">(${file.candidate_name})</span>` : ''}
                            </div>
                        `).join('')}
                    </div>
                    
                    <div class="next-step">
                        <p>🎯 Files have been uploaded and data extracted. Click "Start Analysis" to begin candidate assessment.</p>
                    </div>
                </div>
            `;
        }
    },

    // Display analysis results
    displayAnalysisResults(data) {
        const resultsListEl = document.getElementById('rankingResults');
        if (resultsListEl) {
            const results = data.results.assessments || [];
            
            if (results.length === 0) {
                resultsListEl.innerHTML = '<p>No analysis results available.</p>';
                return;
            }

            // Sort by score descending
            results.sort((a, b) => (b.score || 0) - (a.score || 0));

            resultsListEl.innerHTML = `
                <div class="analysis-results">
                    <h4>🎯 Assessment Results</h4>
                    <div class="results-list">
                        ${results.map((result, index) => `
                            <div class="result-item">
                                <div class="rank-badge">#${index + 1}</div>
                                <div class="candidate-info">
                                    <h5>${result.name}</h5>
                                    <div class="score-container">
                                        <span class="score">${result.score}%</span>
                                        <span class="recommendation ${result.recommendation}">${result.recommendation}</span>
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                    
                    <div class="completion-info">
                        <p>✅ Assessment completed! ${data.results.completed} candidates assessed successfully.</p>
                        <p>📊 Results are now available in the Applicants section.</p>
                    </div>
                </div>
            `;
        }
    },

    // Show analysis button (Step 1 complete)
    showAnalysisButton() {
        const uploadBtnIds = ['resumeStartUploadBtn', 'startUploadBtn'];
        for (const id of uploadBtnIds) {
            const uploadBtn = document.getElementById(id);
            if (uploadBtn) {
                // Convert upload button to analysis button
                const span = uploadBtn.querySelector('span');
                const loader = uploadBtn.querySelector('.btn-loader');
                
                if (span) span.textContent = 'Start Analysis';
                if (loader) loader.style.display = 'none';
                
                uploadBtn.style.display = 'block';
                uploadBtn.disabled = false;
                uploadBtn.className = 'btn btn-success btn-lg'; // Change color to indicate different action
                
                // Remove old click listeners and add analysis listener
                const newBtn = uploadBtn.cloneNode(true);
                uploadBtn.parentNode.replaceChild(newBtn, uploadBtn);
                
                newBtn.addEventListener('click', () => this.startAnalysis());
                
                break;
            }
        }
    },

    // Hide analysis button
    hideAnalysisButton() {
        const uploadBtnIds = ['resumeStartUploadBtn', 'startUploadBtn'];
        for (const id of uploadBtnIds) {
            const uploadBtn = document.getElementById(id);
            if (uploadBtn) {
                uploadBtn.style.display = 'none';
                break;
            }
        }
    },

    // Show completion message
    showCompletionMessage() {
        if (typeof ToastUtils !== 'undefined') {
            ToastUtils.showSuccess('🎉 Upload and analysis complete! Check the Applicants section for detailed results.');
        }
        
        // Add a "New Upload" button to restart the process
        const uploadBtnIds = ['resumeStartUploadBtn', 'startUploadBtn'];
        for (const id of uploadBtnIds) {
            const uploadBtn = document.getElementById(id);
            if (uploadBtn) {
                const span = uploadBtn.querySelector('span');
                const icon = uploadBtn.querySelector('i:not(.btn-loader i)');
                
                if (span) span.textContent = 'Start New Upload';
                if (icon) {
                    icon.className = 'fas fa-plus me-2';
                }
                
                uploadBtn.style.display = 'block';
                uploadBtn.disabled = false;
                uploadBtn.className = 'btn btn-outline-primary btn-lg';
                
                // Remove old listeners and add reset listener
                const newBtn = uploadBtn.cloneNode(true);
                uploadBtn.parentNode.replaceChild(newBtn, uploadBtn);
                newBtn.addEventListener('click', () => this.resetUploadState());
                
                break;
            }
        }
    },

    // Show upload results
    showResults(data) {
        console.log('Upload results:', data);
        
        // Find and show the results step
        const resultStepIds = ['resumeResultsStep', 'resultsStep'];
        let resultsStep = null;
        
        for (const id of resultStepIds) {
            resultsStep = document.getElementById(id);
            if (resultsStep) {
                resultsStep.style.display = 'block';
                break;
            }
        }
        
        if (!resultsStep) {
            console.error('Results step element not found');
            return;
        }
        
        const results = data.results || [];
        const processedCount = results.length;
        const avgScore = results.length > 0 ? 
            Math.round(results.reduce((sum, r) => sum + (r.matchScore || 0), 0) / results.length) : 0;
        const topCandidates = results.filter(r => (r.matchScore || 0) >= 80).length;
        
        // Update summary statistics
        const processedCountEl = document.getElementById('processedCount');
        const avgScoreEl = document.getElementById('avgScore');
        const topCandidatesEl = document.getElementById('topCandidates');
        
        if (processedCountEl) processedCountEl.textContent = processedCount;
        if (avgScoreEl) avgScoreEl.textContent = avgScore + '%';
        if (topCandidatesEl) topCandidatesEl.textContent = topCandidates;
        
        // Display results list
        const resultsListEl = document.getElementById('rankingResults');
        if (resultsListEl) {
            if (results.length === 0) {
                resultsListEl.innerHTML = `
                    <div class="no-results">
                        <i class="fas fa-inbox"></i>
                        <h4>No candidates processed</h4>
                        <p>No valid resumes were found in the uploaded files.</p>
                    </div>
                `;
            } else {
                resultsListEl.innerHTML = results.map((candidate, index) => `
                    <div class="candidate-result-card">
                        <div class="candidate-rank">#${index + 1}</div>
                        <div class="candidate-info">
                            <div class="candidate-header">
                                <h5 class="candidate-name">${this.escapeHtml(candidate.name || candidate.filename)}</h5>
                                <div class="candidate-score">
                                    <span class="score-value">${candidate.matchScore || 0}%</span>
                                    <div class="score-bar">
                                        <div class="score-fill" style="width: ${candidate.matchScore || 0}%"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="candidate-details">
                                <div class="candidate-email">
                                    <i class="fas fa-envelope"></i>
                                    ${this.escapeHtml(candidate.email || 'No email provided')}
                                </div>
                                <div class="candidate-file">
                                    <i class="fas fa-file"></i>
                                    ${this.escapeHtml(candidate.filename)}
                                </div>
                                <div class="candidate-status">
                                    <span class="status-badge status-${candidate.status || 'processed'}">${candidate.status || 'Processed'}</span>
                                </div>
                            </div>
                        </div>
                        <div class="candidate-actions">
                            <button class="btn btn-sm btn-outline-primary" onclick="viewCandidate('${candidate.filename}')">
                                <i class="fas fa-eye"></i> View
                            </button>
                        </div>
                    </div>
                `).join('');
            }
        }
        
        // Show success message
        if (typeof ToastUtils !== 'undefined') {
            ToastUtils.showSuccess(`Successfully processed ${processedCount} files!`);
        }
        
        // Hide upload progress
        this.hideUploadProgress();
        
        // Reset upload state
        this.isUploading = false;
        this.selectedFiles = [];
        this.updateFilePreview();
        this.updateUploadButton();
    },

    // Load LSPU job postings for target position selection
    async loadJobPostings() {
        console.log('🔄 Loading job postings...');
        
        try {
            const response = await fetch('/api/job-postings');
            const data = await response.json();
            
            console.log('📡 API Response:', data);
            
            // Handle different response formats
            let jobPostings;
            if (data.success && Array.isArray(data.postings)) {
                jobPostings = data.postings;
            } else if (Array.isArray(data)) {
                jobPostings = data;
            } else {
                console.error('Unexpected API response format:', data);
                jobPostings = [];
            }
            
            console.log('📋 Job postings to render:', jobPostings.length, jobPostings);
            
            // Store job postings for compatibility
            this.jobs = jobPostings;
            
            if (jobPostings && jobPostings.length > 0) {
                this.renderJobPostings(jobPostings);
            } else {
                console.error('Failed to load job postings or no job postings available');
                this.showNoJobsMessage();
            }
        } catch (error) {
            console.error('Error loading job postings:', error);
            this.showNoJobsMessage();
        }
    },

    // Render LSPU job postings for selection
    renderJobPostings(jobPostings) {
        console.log('🎨 Rendering job postings...');
        
        const positionGrid = document.getElementById('positionTypesUpload');
        if (!positionGrid) {
            console.error('❌ Position types grid not found! Expected element with ID: positionTypesUpload');
            return;
        }
        
        console.log('✅ Found position grid element:', positionGrid);

        if (jobPostings.length === 0) {
            console.log('⚠️ No job postings to render');
            this.showNoJobsMessage();
            return;
        }

        console.log(`📝 Rendering ${jobPostings.length} job postings...`);
        
        positionGrid.innerHTML = jobPostings.map(job => `
            <div class="position-type-card" data-job-id="${job.id}">
                <div class="position-type-header">
                    <h4>${this.escapeHtml(job.title || job.position_title || 'Untitled Position')}</h4>
                    <div class="job-posting-badges">
                        <span class="badge bg-primary">${this.escapeHtml(job.position_type || 'University Position')}</span>
                        <span class="badge bg-info">${this.escapeHtml(job.campus || 'Main Campus')}</span>
                        ${job.status ? `<span class="badge bg-success">${this.escapeHtml(job.status)}</span>` : ''}
                    </div>
                </div>
                <div class="position-type-body">
                    <div class="job-posting-details">
                        <p class="job-reference"><strong>Ref:</strong> ${this.escapeHtml(job.reference_number || 'N/A')}</p>
                        ${job.quantity ? `<p class="job-quantity"><strong>Positions:</strong> ${job.quantity}</p>` : ''}
                        ${job.deadline ? `<p class="job-deadline"><strong>Deadline:</strong> ${new Date(job.deadline).toLocaleDateString()}</p>` : ''}
                    </div>
                </div>
                <div class="position-type-footer">
                    <button class="btn btn-primary select-position" data-job-id="${job.id}" data-job-title="${this.escapeHtml(job.title || job.position_title || 'Untitled Position')}">
                        <i class="fas fa-check-circle me-2"></i>Select Position
                    </button>
                </div>
            </div>
        `).join('');

        // Add click event listeners to the selection buttons
        this.attachPositionSelectionListeners();
    },

    showNoJobsMessage() {
        const positionGrid = document.getElementById('positionTypesUpload');
        if (!positionGrid) return;

        positionGrid.innerHTML = `
            <div class="no-positions-message">
                <div class="no-positions-icon">
                    <i class="fas fa-university"></i>
                </div>
                <h4>No Job Postings Available</h4>
                <p>Please go to the "Job Postings" section and create university job postings first.</p>
                <a href="#job-postings" class="btn btn-primary" onclick="NavigationModule.showSection('job-postings')">
                    <i class="fas fa-plus me-2"></i>Create Job Postings
                </a>
            </div>
        `;
    },

    // Attach event listeners to position selection buttons
    attachPositionSelectionListeners() {
        const positionGrid = document.getElementById('positionTypesUpload');
        if (!positionGrid) return;

        const selectButtons = positionGrid.querySelectorAll('.select-position');
        selectButtons.forEach(button => {
            button.addEventListener('click', () => {
                const jobId = button.getAttribute('data-job-id');
                const jobTitle = button.getAttribute('data-job-title');
                this.selectJob(jobId, jobTitle);
            });
        });
    },

    // Escape HTML to prevent XSS
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    // Select a job posting
    async selectJob(jobId, jobTitle = null) {
        this.selectedJobId = jobId;
        
        // If jobTitle not provided, fetch it from the loaded job postings
        if (!jobTitle && this.jobs && this.jobs.length > 0) {
            const jobPosting = this.jobs.find(j => j.id == jobId);
            jobTitle = jobPosting ? (jobPosting.title || jobPosting.position_title) : `Job Posting ${jobId}`;
        } else if (!jobTitle) {
            // Fallback: fetch job posting data from API
            try {
                const response = await fetch(`/api/job-postings/${jobId}`);
                const data = await response.json();
                if (data.success && data.job_posting) {
                    jobTitle = data.job_posting.position_title || `Job Posting ${jobId}`;
                } else {
                    jobTitle = `Job Posting ${jobId}`;
                }
            } catch (error) {
                console.error('Error fetching job posting:', error);
                jobTitle = `Job Posting ${jobId}`;
            }
        }
        
        // Update UI to show selected job
        const selectedJobTitle = document.getElementById('selectedJobTitle');
        if (selectedJobTitle) {
            selectedJobTitle.textContent = jobTitle;
        }

        // Fetch detailed job posting information
        let jobPostingDetails = null;
        try {
            const response = await fetch(`/api/job-postings/${jobId}`);
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    jobPostingDetails = data.job_posting;
                }
            }
        } catch (error) {
            console.warn('Could not fetch job posting details:', error);
        }

        // Show selected job details with requirements
        const selectedJobDetails = document.getElementById('selectedPositionInfo');
        if (selectedJobDetails) {
            selectedJobDetails.style.display = 'block';
            
            // Update the job title in the preview
            const jobTitleElement = selectedJobDetails.querySelector('.position-title');
            if (jobTitleElement) {
                jobTitleElement.textContent = jobTitle;
            }

            // Extract job requirements from job posting details
            const jobRequirements = jobPostingDetails ? {
                education: jobPostingDetails.education_requirements || '',
                experience: jobPostingDetails.experience_requirements || '',
                skills: jobPostingDetails.skills_requirements || '',
                qualifications: jobPostingDetails.preferred_qualifications || '',
                position_category: jobPostingDetails.position_category || ''
            } : {};

            // Update requirements display if container exists
            const requirementsContainer = selectedJobDetails.querySelector('.job-requirements-preview');
            if (requirementsContainer) {
                this.displayJobRequirements(requirementsContainer, jobRequirements);
            }
        }

        // Update job selection display for current mode
        this.updateJobSelectionDisplay();
        
        // Update upload button state
        this.updateUploadButton();
    },

    // Display job requirements in the upload section
    displayJobRequirements(container, requirements) {
        if (!requirements) {
            container.innerHTML = `
                <div class="requirements-warning">
                    <i class="fas fa-exclamation-triangle text-warning"></i>
                    <span>No detailed requirements set for this position. Assessment may be limited.</span>
                </div>
            `;
            return;
        }

        const reqHtml = `
            <div class="requirements-preview">
                <h6><i class="fas fa-clipboard-check"></i> Position Requirements</h6>
                
                ${requirements.minimum_education ? `
                    <div class="requirement-item">
                        <i class="fas fa-graduation-cap text-primary"></i>
                        <strong>Education:</strong> ${requirements.minimum_education}
                    </div>
                ` : ''}
                
                ${requirements.required_experience ? `
                    <div class="requirement-item">
                        <i class="fas fa-briefcase text-success"></i>
                        <strong>Experience:</strong> ${requirements.required_experience} years
                    </div>
                ` : ''}
                
                ${requirements.subject_area ? `
                    <div class="requirement-item">
                        <i class="fas fa-book text-info"></i>
                        <strong>Field:</strong> ${requirements.subject_area}
                    </div>
                ` : ''}
                
                ${requirements.required_skills && requirements.required_skills.length > 0 ? `
                    <div class="requirement-item skills-preview">
                        <i class="fas fa-cogs text-secondary"></i>
                        <strong>Required Skills:</strong>
                        <div class="skills-tags-mini">
                            ${requirements.required_skills.map(skill => `
                                <span class="skill-tag mini">${skill}</span>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
                
                ${requirements.required_certifications && requirements.required_certifications.length > 0 ? `
                    <div class="requirement-item certifications-preview">
                        <i class="fas fa-certificate text-warning"></i>
                        <strong>Certifications:</strong>
                        <div class="certifications-tags-mini">
                            ${requirements.required_certifications.map(cert => `
                                <span class="certification-tag mini">${cert}</span>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;

        container.innerHTML = reqHtml;
    },

    updateJobSelectionDisplay() {
        if (!this.selectedJobId) return;
        
        // Get job title from stored jobs data
        let jobTitle = 'Selected Job';
        if (this.jobs && this.jobs.length > 0) {
            const job = this.jobs.find(j => j.id == this.selectedJobId);
            jobTitle = job ? job.title : `Job ${this.selectedJobId}`;
        }
        
        // Show upload zones based on current mode
        const regularZone = document.getElementById('regularUploadZone');
        const ocrZone = document.getElementById('ocrUploadZone');
        
        if (this.uploadMode === 'ocr' && ocrZone) {
            ocrZone.style.display = 'block';
        } else if (regularZone) {
            regularZone.style.display = 'block';
        }
        
        // Hide upload instructions for both modes
        const regularInstructions = document.getElementById('uploadInstructions');
        const ocrInstructions = document.getElementById('ocrUploadInstructions');
        
        if (regularInstructions) {
            regularInstructions.style.display = 'none';
        }
        if (ocrInstructions) {
            ocrInstructions.style.display = 'none';
        } else {
            console.log('OCR upload instructions element not found - this is expected');
        }
        
        // Show selected job info - use the actual element ID from HTML
        const selectedPositionInfo = document.getElementById('selectedPositionInfo');
        
        if (selectedPositionInfo) {
            selectedPositionInfo.style.display = 'block';
            // Update the position title
            const positionTitle = selectedPositionInfo.querySelector('.position-title');
            if (positionTitle) {
                positionTitle.textContent = jobTitle;
            }
        }
        
        console.log('Job selection display updated:', {
            jobId: this.selectedJobId,
            jobTitle: jobTitle,
            uploadMode: this.uploadMode,
            regularZoneVisible: regularZone ? regularZone.style.display : 'not found',
            ocrZoneVisible: ocrZone ? ocrZone.style.display : 'not found'
        });
    },

    // Load jobs for select dropdown
    async loadJobsForSelect(selectElement) {
        try {
            const response = await fetch('/api/job-postings');
            const data = await response.json();
            
            if (data.success && selectElement) {
                selectElement.innerHTML = '<option value="">Select a job...</option>' +
                    data.postings.map(job => 
                        `<option value="${job.id}">${this.escapeHtml(job.position_title)}</option>`
                    ).join('');
            }
        } catch (error) {
            console.error('Error loading jobs for select:', error);
        }
    },

    // Update progress message during processing
    updateProgressMessage(message) {
        const uploadBtnIds = ['resumeStartUploadBtn', 'startUploadBtn'];
        for (const id of uploadBtnIds) {
            const uploadBtn = document.getElementById(id);
            if (uploadBtn) {
                const span = uploadBtn.querySelector('span');
                if (span) span.textContent = message;
                break;
            }
        }
    },

    // Show final results (combined upload + analysis complete)
    showFinalResults(data) {
        console.log('Final analysis results:', data);

        // Find and show the results step
        const resultStepIds = ['resumeResultsStep', 'resultsStep'];
        let resultsStep = null;
        
        for (const id of resultStepIds) {
            resultsStep = document.getElementById(id);
            if (resultsStep) {
                resultsStep.style.display = 'block';
                resultsStep.scrollIntoView({ behavior: 'smooth' });
                break;
            }
        }

        // Update summary statistics
        const processedCount = data.total_candidates || 0;
        const avgScore = data.average_score || 0;
        const topCandidates = data.high_scorers || 0;

        const processedCountEl = document.getElementById('processedCount');
        const avgScoreEl = document.getElementById('avgScore');
        const topCandidatesEl = document.getElementById('topCandidates');
        
        if (processedCountEl) processedCountEl.textContent = processedCount;
        if (avgScoreEl) avgScoreEl.textContent = Math.round(avgScore) + '%';
        if (topCandidatesEl) topCandidatesEl.textContent = topCandidates;

        // Display final results
        if (data.results && data.results.assessments) {
            this.displayAnalysisResults(data);
        }

        // Hide upload button and show completion
        this.hideUploadButton();
        this.showCompletionMessage();
    },

    // Hide upload button after completion
    hideUploadButton() {
        const uploadBtnIds = ['resumeStartUploadBtn', 'startUploadBtn'];
        for (const id of uploadBtnIds) {
            const uploadBtn = document.getElementById(id);
            if (uploadBtn) {
                uploadBtn.style.display = 'none';
                break;
            }
        }
    },

    // Show completion message
    showCompletionMessage() {
        // Could add a completion banner or message here if needed
        console.log('Analysis completed successfully');
    },

    // Show complete results (single-step processing)
    showCompleteResults(data) {
        console.log('Complete processing results:', data);

        // Find and show the results step
        const resultStepIds = ['resumeResultsStep', 'resultsStep'];
        let resultsStep = null;
        
        for (const id of resultStepIds) {
            resultsStep = document.getElementById(id);
            if (resultsStep) {
                resultsStep.style.display = 'block';
                resultsStep.scrollIntoView({ behavior: 'smooth' });
                break;
            }
        }

        // Update summary statistics
        const results = data.results || [];
        const processedCount = results.length;
        const scores = results.map(r => r.total_score || r.matchScore || 0);
        const avgScore = scores.length > 0 ? 
            Math.round(scores.reduce((sum, score) => sum + score, 0) / scores.length) : 0;
        const topCandidates = results.filter(r => (r.total_score || r.matchScore || 0) >= 70).length;

        const processedCountEl = document.getElementById('processedCount');
        const avgScoreEl = document.getElementById('avgScore');
        const topCandidatesEl = document.getElementById('topCandidates');
        
        if (processedCountEl) processedCountEl.textContent = processedCount;
        if (avgScoreEl) avgScoreEl.textContent = avgScore + '%';
        if (topCandidatesEl) topCandidatesEl.textContent = topCandidates;

        // Display results using existing method
        this.displayAnalysisResults(data);

        // Hide upload button and show completion
        this.hideUploadButton();
        this.showCompletionMessage();
    }
};

// Make globally available
window.UploadModule = UploadModule;

// Global functions for HTML compatibility
window.triggerFileUpload = function() {
    UploadModule.triggerFileUpload();
};

window.clearJobSelection = function() {
    UploadModule.selectedJobId = null;
    
    // Hide selected job/position details (support both old and new IDs)
    const selectedJobDetails = document.getElementById('selectedPositionDetails') || document.getElementById('selectedJobDetails');
    if (selectedJobDetails) {
        selectedJobDetails.style.display = 'none';
    }
    
    // Hide upload zone and show instructions
    const uploadZone = document.getElementById('uploadZone');
    const uploadInstructions = document.getElementById('uploadInstructions');
    
    if (uploadZone) uploadZone.style.display = 'none';
    if (uploadInstructions) uploadInstructions.style.display = 'block';
    
    // Clear any selected files
    UploadModule.clearFiles();
};

// Clear position selection (new terminology)
window.clearPositionSelection = function() {
    return window.clearJobSelection();
};

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    UploadModule.init();
});

// Global function for switching upload modes (called from HTML)
function switchUploadMode(mode) {
    if (typeof UploadModule !== 'undefined' && UploadModule.switchUploadMode) {
        UploadModule.switchUploadMode(mode);
    }
}

// Global function for triggering file upload (called from HTML)
function triggerFileUpload(mode) {
    if (typeof UploadModule !== 'undefined' && UploadModule.triggerFileUpload) {
        UploadModule.triggerFileUpload(mode);
    }
}

// Make UploadModule available globally
window.UploadModule = UploadModule;

// Export for ES6 modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UploadModule;
}