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
        }
    },

    // Setup event listeners
    setupEventListeners() {
        this.navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = link.getAttribute('data-section');
                if (section) {
                    this.showSection(section);
                }
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
