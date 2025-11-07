#!/usr/bin/env python3
"""
Script to fix the duplicate function definitions in app.py
"""

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line where the duplicate section starts
duplicate_start = None
for i, line in enumerate(lines):
    if '# =============================================================================' in line and i > 3000:
        duplicate_start = i
        break

if duplicate_start:
    print(f"Found duplicate section starting at line {duplicate_start + 1}")
    
    # Keep only the lines before the duplicate
    lines = lines[:duplicate_start]
    
    # Add the proper ending
    lines.append("    \n")
    lines.append("    # Production ready settings for cloud deployment\n")
    lines.append("    app.run(host='0.0.0.0', port=port, debug=False)\n")
    
    # Write the fixed file
    with open('app.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Fixed app.py - removed duplicate section, file now has {len(lines)} lines")
else:
    print("No duplicate section found")