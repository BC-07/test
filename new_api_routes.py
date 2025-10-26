#!/usr/bin/env python3
"""
New API endpoints for enhanced PDS processing workflow
This adds the analysis functionality and integrates with our working extraction system
"""

# Add these new routes to the existing app.py _register_routes method:

# Enhanced PDS Upload and Analysis routes
self.app.add_url_rule('/api/upload-pds-enhanced', 'upload_pds_enhanced', self.upload_pds_enhanced, methods=['POST'])
self.app.add_url_rule('/api/start-analysis', 'start_analysis', self.start_analysis, methods=['POST'])
self.app.add_url_rule('/api/analysis-status/<batch_id>', 'get_analysis_status', self.get_analysis_status, methods=['GET'])
self.app.add_url_rule('/api/clear-old-candidates', 'clear_old_candidates', self.clear_old_candidates, methods=['POST'])

# Batch processing routes
self.app.add_url_rule('/api/upload-batches', 'get_upload_batches', self.get_upload_batches, methods=['GET'])
self.app.add_url_rule('/api/upload-batches/<batch_id>', 'get_batch_details', self.get_batch_details, methods=['GET'])

# Enhanced candidates API (replaces existing get_candidates)
self.app.add_url_rule('/api/candidates-enhanced', 'get_candidates_enhanced', self.get_candidates_enhanced, methods=['GET'])