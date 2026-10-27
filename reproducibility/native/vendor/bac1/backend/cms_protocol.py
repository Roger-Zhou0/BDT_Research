"""Isolated copy of the two CMS1 constants used by cms_geometry.
Original full cms_protocol.py SHA: e941b4274b27b1642e9467fdafe144fa79c378c199d2b5d35358fb7428261b7c.
This is NOT the full CMS1 scientific protocol and does not change that study.
"""
VERSION = 'correspondence_matched_s1_v1'
GEOMETRY = dict(zero_angle_sine=1e-7,minimum_candidate_sine=1e-7,
 norm_relative_tolerance=1e-5,common_relative_tolerance=1e-5,
 angle_absolute_tolerance=5e-6,displacement_absolute_tolerance=2e-6,
 displacement_relative_tolerance=.0002)
