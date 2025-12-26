import os
import re
import json

# Configuration
ROOT_DIR = os.path.abspath(".")
EXCLUDE_DIRS = {'.git', '.gradle', 'build', '.idea', '.cxx', 'captures', 'logs'}
EXCLUDE_FILES = {"local.properties", "analyze_paths.py", "lint-results-debug.txt", "mcp-puppeteer-2025-12-24.log"}
# Known safe system paths or patterns to ignore
SAFE_PATTERNS = [
    r"^/usr/bin/env",
    r"^/bin/bash",
    r"^/bin/sh",
    r"^/proc/",
    r"^/sys/",
    r"^/dev/",
    r"https?://",  # Network URLs are not file paths
    r"schemas.android.com",
]

# Patterns for absolute paths
# UNIX: Starts with / and has at least two segments, or /home/, /opt/, /usr/
# WINDOWS: C:\ or D:\
PATH_REGEX = re.compile(r'(?:[\"\"])?((?:/[a-zA-Z0-9_.-]+){2,}|[a-zA-Z]:\\[a-zA-Z0-9_.-]+)(?:[\"\"])?')

def is_safe(path_str):
    for safe in SAFE_PATTERNS:
        if re.search(safe, path_str):
            return True
    return False

def analyze_file(file_path):
    findings = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                matches = PATH_REGEX.findall(line)
                for match in matches:
                    # Filter out simple matches that might be partial URLs or safe
                    if is_safe(match):
                        continue
                    
                    # Heuristics for false positives (e.g., XML namespaces, simple mime types)
                    if match.startswith("//"): continue # Comments or protocol relative
                    if "text/html" in match: continue
                    if "application/json" in match: continue
                    
                    # Check if it looks like a User path
                    severity = "LOW"
                    path_type = "FILE_SYSTEM"
                    
                    if "/home/" in match or "/Users/" in match:
                        severity = "HIGH" # PII / User specific
                    elif "/opt/" in match or "/usr/local/" in match:
                        severity = "MEDIUM" # Environment specific
                    else:
                        severity = "LOW" # Generic system path

                    findings.append({
                        "file": os.path.relpath(file_path, ROOT_DIR),
                        "line": i + 1,
                        "content": line.strip(),
                        "match": match,
                        "type": path_type,
                        "severity": severity
                    })
    except Exception as e:
        pass
        # print(f"Error reading {file_path}: {e}")
    return findings

def main():
    report = []
    for root, dirs, files in os.walk(ROOT_DIR):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            if file in EXCLUDE_FILES:
                continue
            
            file_path = os.path.join(root, file)
            # Skip binary files if possible (simple heuristic)
            if file.endswith(('.png', '.jpg', '.jar', '.apk', '.zip')):
                continue

            findings = analyze_file(file_path)
            report.extend(findings)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
