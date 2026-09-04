import os
import ast
from pathlib import Path

src = Path(r'C:\Bahtiyorjon\QuantAI\src')
test_dir = Path(r'C:\Bahtiyorjon\QuantAI\tests')

src_files = list(Path('src').rglob('*.py'))
test_files = list(Path('tests').rglob('test_*.py'))

print(f'Source files: {len(list(Path("src").rglob("*.py")))}')
print(f'Test files: {len(list(Path("tests").rglob("test_*.py")))}')

# Check for empty files
empty_files = []
for py_file in Path('src').rglob('*.py'):
    if Path(py_file).stat().st_size == 0:
        print(f'EMPTY: {py_file.relative_to(Path("src"))}')

# Check test files
for py_file in Path('tests').rglob('test_*.py'):
    if Path(py_file).stat().st_size == 0:
        print(f'EMPTY TEST: {py_file.relative_to(Path("tests"))}')

# Check imports
import ast
for py_file in Path('src').rglob('*.py'):
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('src.'):
                        print(f'Internal import in {py_file.name}: {alias.name}')
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith('src.'):
                    print(f'Relative import in {py_file.name}: from {node.module} import ...')
    except:
        pass