/* Job Export and Print Utilities */

const JobExportUtils = {
    // Initialize PDF library (jsPDF will be loaded externally)
    init() {
        // Load jsPDF if not already loaded
        if (typeof window.jsPDF === 'undefined') {
            this.loadJsPDF();
        }
    },

    // Load jsPDF library dynamically
    async loadJsPDF() {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js';
            script.onload = () => {
                console.log('jsPDF loaded successfully');
                resolve();
            };
            script.onerror = () => {
                console.error('Failed to load jsPDF');
                reject(new Error('Failed to load PDF library'));
            };
            document.head.appendChild(script);
        });
    },

    // Generate PDF for job posting
    async generateJobPostingPDF(job) {
        try {
            // Ensure jsPDF is loaded
            if (typeof window.jsPDF === 'undefined') {
                await this.loadJsPDF();
            }

            const { jsPDF } = window.jspdf;
            const doc = new jsPDF('p', 'mm', 'a4');
            
            // Page setup
            const pageWidth = doc.internal.pageSize.getWidth();
            const pageHeight = doc.internal.pageSize.getHeight();
            const margin = 20;
            const contentWidth = pageWidth - (margin * 2);
            let yPosition = margin;

            // Company header
            doc.setFillColor(37, 99, 235); // Primary color
            doc.rect(0, 0, pageWidth, 25, 'F');
            
            doc.setTextColor(255, 255, 255);
            doc.setFontSize(18);
            doc.setFont(undefined, 'bold');
            doc.text('Company Name', margin, 15);
            
            doc.setFontSize(12);
            doc.setFont(undefined, 'normal');
            doc.text('Job Posting', pageWidth - margin - 30, 15);

            yPosition = 40;

            // Job title
            doc.setTextColor(37, 99, 235);
            doc.setFontSize(24);
            doc.setFont(undefined, 'bold');
            doc.text(job.title, margin, yPosition);
            yPosition += 15;

            // Department and basic info
            doc.setTextColor(100, 116, 139);
            doc.setFontSize(12);
            doc.setFont(undefined, 'normal');
            doc.text(`Department: ${job.department}`, margin, yPosition);
            doc.text(`Category: ${job.category}`, margin + 80, yPosition);
            yPosition += 8;
            doc.text(`Experience Level: ${this.formatExperienceLevel(job.experience_level)}`, margin, yPosition);
            doc.text(`Posted: ${new Date().toLocaleDateString()}`, margin + 80, yPosition);
            yPosition += 20;

            // Divider line
            doc.setDrawColor(226, 232, 240);
            doc.setLineWidth(0.5);
            doc.line(margin, yPosition, pageWidth - margin, yPosition);
            yPosition += 15;

            // Job description section
            doc.setTextColor(30, 41, 59);
            doc.setFontSize(14);
            doc.setFont(undefined, 'bold');
            doc.text('Job Description', margin, yPosition);
            yPosition += 10;

            doc.setFontSize(11);
            doc.setFont(undefined, 'normal');
            const descriptionLines = doc.splitTextToSize(job.description, contentWidth);
            doc.text(descriptionLines, margin, yPosition);
            yPosition += (descriptionLines.length * 5) + 15;

            // Required skills section
            if (yPosition > pageHeight - 60) {
                doc.addPage();
                yPosition = margin;
            }

            doc.setFontSize(14);
            doc.setFont(undefined, 'bold');
            doc.text('Required Skills & Qualifications', margin, yPosition);
            yPosition += 10;

            const skills = job.requirements.split(',').map(s => s.trim()).filter(Boolean);
            doc.setFontSize(11);
            doc.setFont(undefined, 'normal');

            skills.forEach((skill, index) => {
                if (yPosition > pageHeight - 30) {
                    doc.addPage();
                    yPosition = margin;
                }
                doc.text(`• ${skill}`, margin + 5, yPosition);
                yPosition += 6;
            });

            yPosition += 15;

            // Application instructions
            if (yPosition > pageHeight - 40) {
                doc.addPage();
                yPosition = margin;
            }

            doc.setFontSize(14);
            doc.setFont(undefined, 'bold');
            doc.text('How to Apply', margin, yPosition);
            yPosition += 10;

            doc.setFontSize(11);
            doc.setFont(undefined, 'normal');
            const applicationText = 'yadayayadadayaydsaydasydasd';
            const applicationLines = doc.splitTextToSize(applicationText, contentWidth);
            doc.text(applicationLines, margin, yPosition);
            yPosition += (applicationLines.length * 5) + 20;

            // Footer
            doc.setDrawColor(226, 232, 240);
            doc.line(margin, pageHeight - 30, pageWidth - margin, pageHeight - 30);
            
            doc.setTextColor(148, 163, 184);
            doc.setFontSize(9);
            doc.text('Generated by Resume Screening AI', margin, pageHeight - 20);
            doc.text(`Generated on ${new Date().toLocaleString()}`, pageWidth - margin - 50, pageHeight - 20);

            return doc;

        } catch (error) {
            console.error('Error generating PDF:', error);
            throw new Error('Failed to generate PDF');
        }
    },

    // Format experience level for display
    formatExperienceLevel(level) {
        const levels = {
            'entry': 'Entry Level (0-2 years)',
            'mid': 'Mid Level (2-5 years)',
            'senior': 'Senior Level (5+ years)',
            'lead': 'Lead/Manager (7+ years)'
        };
        return levels[level] || level;
    },

    // Export job as PDF
    async exportJobAsPDF(job) {
        try {
            const doc = await this.generateJobPostingPDF(job);
            const fileName = `${job.title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_job_posting.pdf`;
            doc.save(fileName);
            
            ToastUtils.showSuccess('Job posting PDF downloaded successfully');
        } catch (error) {
            console.error('Error exporting PDF:', error);
            ToastUtils.showError('Failed to export PDF: ' + error.message);
        }
    },

    // Print job posting
    async printJobPosting(job) {
        try {
            const doc = await this.generateJobPostingPDF(job);
            
            // Open PDF in new window for printing
            const pdfBlob = doc.output('blob');
            const url = URL.createObjectURL(pdfBlob);
            
            const printWindow = window.open(url, '_blank');
            printWindow.onload = () => {
                printWindow.print();
                // Clean up the URL after a delay
                setTimeout(() => {
                    URL.revokeObjectURL(url);
                }, 1000);
            };
            
            ToastUtils.showSuccess('Opening print dialog...');
        } catch (error) {
            console.error('Error printing job posting:', error);
            ToastUtils.showError('Failed to print job posting: ' + error.message);
        }
    },

    // Generate simple HTML preview for quick viewing
    generateHTMLPreview(job) {
        return `
            <div style="font-family: 'Inter', Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px 20px; background: #fff; color: #1e293b;">
                <!-- Header -->
                <div style="background: linear-gradient(135deg, #2563eb, #3b82f6); color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px;">
                    <h1 style="margin: 0 0 10px 0; font-size: 28px; font-weight: 700;">${job.title}</h1>
                    <div style="display: flex; gap: 30px; font-size: 14px; opacity: 0.9;">
                        <span><strong>Department:</strong> ${job.department}</span>
                        <span><strong>Category:</strong> ${job.category}</span>
                        <span><strong>Experience:</strong> ${this.formatExperienceLevel(job.experience_level)}</span>
                    </div>
                </div>

                <!-- Job Description -->
                <div style="margin-bottom: 30px;">
                    <h2 style="color: #2563eb; font-size: 20px; font-weight: 600; margin-bottom: 15px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;">
                        Job Description
                    </h2>
                    <p style="line-height: 1.6; color: #64748b; font-size: 16px;">${job.description}</p>
                </div>

                <!-- Required Skills -->
                <div style="margin-bottom: 30px;">
                    <h2 style="color: #2563eb; font-size: 20px; font-weight: 600; margin-bottom: 15px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;">
                        Required Skills & Qualifications
                    </h2>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 15px;">
                        ${job.requirements.split(',').map(skill => 
                            `<span style="background: #bae6fd; color: #0369a1; padding: 6px 12px; border-radius: 20px; font-size: 14px; font-weight: 500;">${skill.trim()}</span>`
                        ).join('')}
                    </div>
                </div>

                <!-- Application Instructions -->
                <div style="background: #f8fafc; padding: 25px; border-radius: 8px; border-left: 4px solid #2563eb;">
                    <h3 style="color: #2563eb; font-size: 18px; font-weight: 600; margin-bottom: 10px;">How to Apply</h3>
                    <p style="color: #64748b; line-height: 1.6; margin: 0;">
                        Please submit your resume and cover letter through our online application system. 
                        We are an equal opportunity employer committed to diversity and inclusion.
                    </p>
                </div>

                <!-- Footer -->
                <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0; text-align: center; color: #94a3b8; font-size: 12px;">
                    Generated by Resume Screening AI on ${new Date().toLocaleString()}
                </div>
            </div>
        `;
    },

    // Show HTML preview in modal
    showJobPreview(job) {
        const previewHTML = this.generateHTMLPreview(job);
        
        // Create preview modal if it doesn't exist
        if (!document.getElementById('jobPreviewModal')) {
            const modalHTML = `
                <div class="modal fade" id="jobPreviewModal" tabindex="-1">
                    <div class="modal-dialog modal-xl">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title">Job Posting Preview</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                            </div>
                            <div class="modal-body" style="max-height: 70vh; overflow-y: auto;">
                                <div id="jobPreviewContent"></div>
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-outline-primary" id="printPreviewBtn">
                                    <i class="fas fa-print me-2"></i>Print
                                </button>
                                <button type="button" class="btn btn-primary" id="downloadPreviewBtn">
                                    <i class="fas fa-download me-2"></i>Download PDF
                                </button>
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            document.body.insertAdjacentHTML('beforeend', modalHTML);
        }

        // Update content and show modal
        document.getElementById('jobPreviewContent').innerHTML = previewHTML;
        
        // Update button event listeners
        document.getElementById('printPreviewBtn').onclick = () => this.printJobPosting(job);
        document.getElementById('downloadPreviewBtn').onclick = () => this.exportJobAsPDF(job);
        
        BootstrapInit.showModal('jobPreviewModal');
    },

    // Export all jobs as a single PDF
    async exportAllJobsAsPDF(jobs) {
        try {
            // Ensure jsPDF is loaded
            if (typeof window.jsPDF === 'undefined') {
                await this.loadJsPDF();
            }

            const { jsPDF } = window.jspdf;
            const doc = new jsPDF('p', 'mm', 'a4');
            
            // Page setup
            const pageWidth = doc.internal.pageSize.getWidth();
            const pageHeight = doc.internal.pageSize.getHeight();
            const margin = 20;
            
            // Cover page
            doc.setFillColor(37, 99, 235);
            doc.rect(0, 0, pageWidth, pageHeight, 'F');
            
            doc.setTextColor(255, 255, 255);
            doc.setFontSize(32);
            doc.setFont(undefined, 'bold');
            doc.text('Job Postings Catalog', pageWidth / 2, 80, { align: 'center' });
            
            doc.setFontSize(16);
            doc.setFont(undefined, 'normal');
            doc.text('Resume Screening AI', pageWidth / 2, 100, { align: 'center' });
            doc.text(`${jobs.length} Available Positions`, pageWidth / 2, 120, { align: 'center' });
            doc.text(`Generated on ${new Date().toLocaleDateString()}`, pageWidth / 2, 140, { align: 'center' });

            // Table of contents
            doc.addPage();
            doc.setTextColor(37, 99, 235);
            doc.setFontSize(20);
            doc.setFont(undefined, 'bold');
            doc.text('Table of Contents', margin, 30);
            
            let tocY = 50;
            doc.setFontSize(12);
            doc.setFont(undefined, 'normal');
            
            jobs.forEach((job, index) => {
                if (tocY > pageHeight - 30) {
                    doc.addPage();
                    tocY = 30;
                }
                doc.text(`${index + 1}. ${job.title}`, margin + 5, tocY);
                doc.text(`${job.department}`, margin + 100, tocY);
                tocY += 8;
            });

            // Add each job on separate pages
            for (let i = 0; i < jobs.length; i++) {
                const job = jobs[i];
                doc.addPage();
                
                // Use the same template as single job export
                const tempDoc = await this.generateJobPostingPDF(job);
                const pages = tempDoc.internal.pages;
                
                // Copy pages (skip the first empty page)
                for (let j = 1; j < pages.length; j++) {
                    if (j > 1) doc.addPage();
                    // Note: This is a simplified approach. In production, you'd want to properly copy page content
                    // For now, we'll generate each job individually
                }
                
                // Generate job content directly
                await this.addJobToDocument(doc, job, i + 1);
            }

            const fileName = `all_job_postings_${new Date().toISOString().split('T')[0]}.pdf`;
            doc.save(fileName);
            
            ToastUtils.showSuccess(`${jobs.length} job postings exported successfully`);
        } catch (error) {
            console.error('Error exporting all jobs:', error);
            throw new Error('Failed to export all jobs');
        }
    },

    // Add individual job to document
    async addJobToDocument(doc, job, jobNumber) {
        const pageWidth = doc.internal.pageSize.getWidth();
        const pageHeight = doc.internal.pageSize.getHeight();
        const margin = 20;
        const contentWidth = pageWidth - (margin * 2);
        let yPosition = margin;

        // Job number header
        doc.setFillColor(37, 99, 235);
        doc.rect(0, 0, pageWidth, 25, 'F');
        
        doc.setTextColor(255, 255, 255);
        doc.setFontSize(14);
        doc.setFont(undefined, 'bold');
        doc.text(`Job ${jobNumber}`, margin, 15);
        doc.text(`Page ${doc.internal.getCurrentPageInfo().pageNumber}`, pageWidth - margin - 20, 15);

        yPosition = 40;

        // Job title
        doc.setTextColor(37, 99, 235);
        doc.setFontSize(20);
        doc.setFont(undefined, 'bold');
        doc.text(job.title, margin, yPosition);
        yPosition += 12;

        // Basic info
        doc.setTextColor(100, 116, 139);
        doc.setFontSize(10);
        doc.setFont(undefined, 'normal');
        doc.text(`${job.department} • ${job.category} • ${this.formatExperienceLevel(job.experience_level)}`, margin, yPosition);
        yPosition += 15;

        // Description
        doc.setTextColor(30, 41, 59);
        doc.setFontSize(11);
        const descLines = doc.splitTextToSize(job.description, contentWidth);
        doc.text(descLines, margin, yPosition);
        yPosition += (descLines.length * 4) + 10;

        // Skills
        doc.setFontSize(10);
        doc.setFont(undefined, 'bold');
        doc.text('Required Skills:', margin, yPosition);
        yPosition += 6;
        
        doc.setFont(undefined, 'normal');
        const skills = job.requirements.split(',').map(s => s.trim()).filter(Boolean);
        skills.forEach(skill => {
            doc.text(`• ${skill}`, margin + 5, yPosition);
            yPosition += 5;
        });
    },
};

// Make globally available
window.JobExportUtils = JobExportUtils;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    JobExportUtils.init();
});
