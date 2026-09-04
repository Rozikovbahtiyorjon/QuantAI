path = r'C:\Bahtiyorjon\QuantAI\src\control_plane\supervisor.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed = []
for line in lines:
    # Fix lines that are inside class methods but have wrong indent
    # If line starts with "self." and has 0 or 4 indent but should be 8, fix
    stripped = line.lstrip()
    if stripped.startswith('self.') or stripped.startswith('evidence =') or stripped.startswith('result =') or stripped.startswith('agent_type =') or stripped.startswith('if ') or stripped.startswith('for ') or stripped.startswith('return ') or stripped.startswith('await ') or stripped.startswith('import '):
        # Check indent
        indent = len(line) - len(stripped)
        # If inside a method, should be at least 8
        # Simple heuristic: if previous line was indented 8 and this is 4 or 0, fix to 8
        # For now, fix known broken lines: those with indent 0 or 4 that should be 8 inside methods
        # Detect by checking if line is inside class (contains "class SignalGenerator" earlier)
        pass
    fixed.append(line)

# Simple fix: for any line that is "self.audit_logger.log(" with indent 4 inside a method, make it 8
# Actually, just fix line 529 and similar: where indent is 4 but should be 8
new_lines = []
in_method = False
for i, line in enumerate(lines):
    stripped = line.lstrip()
    if stripped.startswith('class ') or stripped.startswith('def ') or stripped.startswith('async def '):
        # method def inside class has 4 spaces
        if line.startswith('    def ') or line.startswith('    async def '):
            in_method = True
        elif line.startswith('def ') or line.startswith('async def '):
            in_method = False
        new_lines.append(line)
        continue
    if in_method:
        # Inside method, code should be 8 spaces
        if stripped and not stripped.startswith('#'):
            # Check if this line is incorrectly at 4
            if line.startswith('    self.') or line.startswith('    evidence') or line.startswith('    result') or line.startswith('    agent_type'):
                # Should be 8
                new_lines.append('    ' + line)
                continue
    new_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('fixed')
