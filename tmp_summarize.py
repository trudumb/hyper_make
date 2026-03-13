import sys
from collections import defaultdict
import re

def summarize(filename):
    file_counts = defaultdict(int)
    consts = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        if not line.strip(): continue
        parts = line.split(':', 2)
        if len(parts) < 3: continue
        
        filepath, line_num, content = parts[0], parts[1], parts[2]
        
        # Simplify path
        filepath = filepath.replace('\\', '/')
        if 'src/' in filepath:
            filepath = filepath[filepath.find('src/'):]
            
        file_counts[filepath] += 1
        
        if 'const ' in content or 'static ' in content:
            if 'f64' in content or 'f32' in content:
                consts.append(f"- `{filepath}:{line_num}`: `{content.strip()}`")

    # Generate Report
    report = []
    report.append("# Arbitrary Parameters and Magic Numbers Report")
    report.append("This report outlines the ad hoc multipliers and arbitrary parameters discovered in the codebase.\n")
    
    report.append("## 1. Explicit Parameter Constants (`f64`/`f32`)")
    report.append("These values are hardcoded as constants, acting as arbitrary parameters:")
    
    # Sort consts by filepath
    consts.sort()
    for c in consts:
        report.append(c)
        
    report.append("\n## 2. Files with the Highest Concentration of Magic Numbers")
    report.append("Many parameters and ad hoc multipliers are used inline. Here are the files with the most magic numbers (inline floats / inline arithmetic):")
    
    sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
    for filepath, count in sorted_files[:30]:
        report.append(f"- `{filepath}`: {count} inline instances")
        
    with open('arbitrary_parameters_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

if __name__ == '__main__':
    summarize('tmp_magic_results.txt')

