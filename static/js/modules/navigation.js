// Navigation Module
const NavigationModule = {
    // Initialize navigation functionality
    init() {
        this.navLinks = document.querySelectorAll('.nav-link');
        this.sections = document.querySelectorAll('.content-section');
        this.sectionTitle = document.getElementById('sectionTitle');
        
        this.setupEventListeners();
        this.loadInitialSection();
    },

    // Show a specific section
    showSection(sectionId) {
        // Hide all sections first
        this.sections.forEach(section => {
            section.classList.remove('active');
            section.style.display = 'none';
        });

        // Remove active class from all nav links
        this.navLinks.forEach(link => link.classList.remove('active'));

        // Show the target section
        const targetSection = document.getElementById(`${sectionId}Section`);
        const targetLink = document.querySelector(`[data-section="${sectionId}"]`);

        if (targetSection && targetLink) {
            targetSection.style.display = 'block';
            targetSection.classList.add('active');
            targetLink.classList.add('active');
            
            // Update page title
            if (this.sectionTitle) {
                const titleSpan = targetLink.querySelector('span');
                this.sectionTitle.textContent = titleSpan ? titleSpan.textContent : sectionId;
            }
            
            // Load section-specific data
            this.loadSectionData(sectionId);
        }
    },

    // Load data for specific sections
    loadSectionData(sectionId) {
        switch(sectionId) {
            case 'upload':
                if (typeof UploadModule !== 'undefined' && UploadModule.init) {
                    UploadModule.init();
                }
                if (typeof loadJobCategoriesForUpload === 'function') {
                    loadJobCategoriesForUpload();
                }
                break;
            case 'candidates':
                if (typeof loadCandidatesSection === 'function') {
                    loadCandidatesSection();
                }
                break;
            case 'dashboard':
                if (typeof loadDashboardData === 'function') {
                    loadDashboardData();
                }
                break;
            case 'analytics':
                if (typeof loadAnalytics === 'function') {
                    loadAnalytics();
                }
                break;
            case 'job-postings':
                if (typeof jobPostingManager !== 'undefined' && jobPostingManager.loadJobPostings) {
                    // Show the job posting management section
                    const jobPostingSection = document.getElementById('jobPostingManagement');
                    if (jobPostingSection) {
                        jobPostingSection.style.display = 'block';
                        jobPostingManager.loadJobPostings();
                    }
                }
                break;
            case 'user-management':
                if (typeof UserManagementModule !== 'undefined' && UserManagementModule.loadUsers) {
                    UserManagementModule.loadUsers();
                }
                break;
        }
    },

    // Setup event listeners
    setupEventListeners() {
        this.navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                const section = link.getAttribute('data-section');
                if (section) {
                    // Only prevent default for section-based navigation
                    e.preventDefault();
                    this.showSection(section);
                }
                // Allow normal navigation for links without data-section (like logout, user-management)
            });
        });
    },

    // Load initial section
    loadInitialSection() {
        const initialSection = window.location.hash.slice(1) || 'dashboard';
        this.showSection(initialSection);
    }
};

// Make globally available
window.NavigationModule = NavigationModule;
window.showSection = NavigationModule.showSection.bind(NavigationModule);

// Backward compatibility
window.setupNavigation = NavigationModule.init.bind(NavigationModule);
