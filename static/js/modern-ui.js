/**
 * Modern UI Enhancements - This file adds interactive animations and UI improvements
 */

document.addEventListener('DOMContentLoaded', function() {
    enhanceUI();
    fixDashboardLayout();
});

function enhanceUI() {
    // Add scroll-triggered animations
    setupScrollAnimations();
    
    // Enhance buttons with hover effect
    enhanceButtons();
    
    // Add card hover effects
    enhanceCards();
    
    // Apply custom form styling
    enhanceFormElements();
}

/**
 * Sets up animations that trigger on scroll
 */
function setupScrollAnimations() {
    // Only setup if IntersectionObserver is supported
    if (!('IntersectionObserver' in window)) return;
    
    const observerOptions = {
        threshold: 0.15,
        rootMargin: "0px 0px -50px 0px"
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-fade-in-up');
                // Once the animation has played, no need to observe anymore
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe elements that should animate on scroll but don't have the animation class yet
    document.querySelectorAll('.feature-card, .stats-card, .team-member, .about-feature, .upload-step')
        .forEach(el => {
            if (!el.classList.contains('animate-fade-in-up')) {
                observer.observe(el);
            }
        });
}

/**
 * Fixes dashboard layout issues, especially the scrolling gap
 */
function fixDashboardLayout() {
    // Only run this on dashboard page
    if (!document.querySelector('.dashboard-container')) return;
    
    // Fix the body padding that may cause gaps
    document.body.style.paddingTop = '0';
    document.body.style.overflow = 'hidden';
    
    // Make main content take full available height
    const mainContent = document.querySelector('.main-content');
    if (mainContent) {
        // Ensure proper height calculations
        const updateHeight = () => {
            const windowHeight = window.innerHeight;
            mainContent.style.height = `${windowHeight}px`;
            mainContent.style.overflowY = 'auto';
        };
        
        // Set initial height
        updateHeight();
        
        // Update on resize
        window.addEventListener('resize', updateHeight);
    }
    
    // Enhance job cards with subtle interactions
    document.querySelectorAll('.job-card').forEach(card => {
        // Add subtle shadow increase on hover
        card.addEventListener('mouseenter', function() {
            this.style.boxShadow = 'var(--shadow-card-hover)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.boxShadow = 'var(--shadow-card)';
        });
    });
}

/**
 * Enhances buttons with hover effects and micro-interactions
 */
function enhanceButtons() {
    document.querySelectorAll('.btn').forEach(button => {
        // Add ripple effect to buttons
        button.addEventListener('mousedown', function(e) {
            const ripple = document.createElement('span');
            ripple.classList.add('btn-ripple');
            this.appendChild(ripple);
            
            const rect = button.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            
            ripple.style.width = ripple.style.height = `${size}px`;
            ripple.style.left = `${e.clientX - rect.left - size/2}px`;
            ripple.style.top = `${e.clientY - rect.top - size/2}px`;
            
            ripple.classList.add('active');
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
}

/**
 * Enhances cards with subtle interactions
 */
function enhanceCards() {
    // Only apply hover effects on desktop
    if (window.innerWidth < 768) return;
    
    document.querySelectorAll('.feature-card, .upload-card, .dashboard-card, .stats-card').forEach(card => {
        // Add subtle tilt effect
        card.addEventListener('mousemove', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const tiltX = (centerX - x) / 20;
            const tiltY = (y - centerY) / 10;
            
            this.style.transform = `perspective(1000px) rotateX(${tiltY}deg) rotateY(${-tiltX}deg) scale3d(1.02, 1.02, 1.02)`;
        });
        
        // Reset transform on mouse leave
        card.addEventListener('mouseleave', function() {
            this.style.transform = '';
        });
    });
}

/**
 * Enhances form elements with modern styling and interactions
 */
function enhanceFormElements() {
    // Add floating label effect to form inputs
    document.querySelectorAll('.form-control').forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('input-focused');
        });
        
        input.addEventListener('blur', function() {
            if (this.value === '') {
                this.parentElement.classList.remove('input-focused');
            }
        });
        
        // Initial state check
        if (input.value !== '') {
            input.parentElement.classList.add('input-focused');
        }
    });
    
    // Enhance file uploads
    document.querySelectorAll('.upload-zone').forEach(zone => {
        zone.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('upload-zone-active');
        });
        
        zone.addEventListener('dragleave', function() {
            this.classList.remove('upload-zone-active');
        });
        
        zone.addEventListener('drop', function() {
            this.classList.remove('upload-zone-active');
        });
    });
}

// Add CSS for dynamic effects
(function() {
    const style = document.createElement('style');
    style.textContent = `
        .btn-ripple {
            position: absolute;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.4);
            transform: scale(0);
            animation: ripple 0.6s linear;
            pointer-events: none;
        }
        
        @keyframes ripple {
            to {
                transform: scale(2);
                opacity: 0;
            }
        }
        
        .upload-zone-active {
            border-color: var(--primary-color);
            background-color: rgba(67, 97, 238, 0.05);
        }
        
        .input-focused label {
            transform: translateY(-20px) scale(0.85);
            color: var(--primary-color);
        }
        
        /* Fix for job tags display */
        .job-tags {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .job-category, .job-experience {
            font-size: 0.75rem;
            padding: 0.25rem 0.75rem;
        }
        
        /* Fix for the potential scrolling gap */
        .dashboard-container {
            height: 100vh;
            overflow: hidden;
        }
    `;
    document.head.appendChild(style);
})();
