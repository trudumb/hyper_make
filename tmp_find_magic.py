import os
import re
import sys

def find_magic_numbers(directory):
    pattern = re.compile(r'[^a-zA-Z0-9_](\d+\.\d+(?:e-?\d+)?f?(?:32|64)?)\b')
    const_pattern = re.compile(r'\b(const|static)\s+[A-Z_0-9]+\s*:\s*(f32|f64|usize|isize|i32|u32)\s*=\s*([^;]+);')
    
    results = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith('.rs'):
                continue
                
            filepath = os.path.join(root, file)
            filepath = filepath.replace('\\', '/')
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                continue
                
            for i, line in enumerate(lines):
                original_line = line.strip()
                line = original_line
                if line.startswith('//') or line.startswith('//!') or line.startswith('///'):
                    continue
                
                const_match = const_pattern.search(line)
                if const_match:
                    results.append(f"{filepath}:{i+1}:{original_line}")
                    continue
                    
                if any(op in line for op in ['*', '/', '+', '-', '=', '>', '<']):
                    matches = pattern.findall(line)
                    for m in set(matches):
                        if m in ['0.0', '1.0', '0.0f64', '1.0f64', '0.0f32', '1.0f32', '24.0', '60.0']:
                            continue
                        results.append(f"{filepath}:{i+1}:{original_line}")
                        break

    return results

if __name__ == '__main__':
    res = find_magic_numbers(sys.argv[1])
    with open('tmp_magic_results.txt', 'w', encoding='utf-8') as f:
        for r in res:
            f.write(r + '\n')
    print(f"Wrote {len(res)} matches to tmp_magic_results.txt")
