"""
tools/rerender_check.py
=======================
Prints every table header row and every line containing '$' in reports/phase4_summary.md
to visually verify that all LaTeX math symbols, Greek letters, and headers are pristine.
"""

from pathlib import Path
import re

def main():
    p = Path("reports/phase4_summary.md")
    lines = p.read_text(encoding="utf-8").splitlines()

    print("================================================================================")
    print("  TABLE HEADER ROWS IN reports/phase4_summary.md")
    print("================================================================================")
    for i, line in enumerate(lines, 1):
        if line.startswith("|") and i < len(lines):
            next_line = lines[i].strip()
            if re.match(r"^\|(?:\s*[:\-]+\s*\|)+$", next_line):
                print(f"Line {i:3d}: {line}")

    print("\n================================================================================")
    print("  ALL LINES CONTAINING '$' IN reports/phase4_summary.md")
    print("================================================================================")
    for i, line in enumerate(lines, 1):
        if "$" in line:
            print(f"Line {i:3d}: {line}")

if __name__ == "__main__":
    main()
