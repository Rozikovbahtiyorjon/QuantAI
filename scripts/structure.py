from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
TESTS_DIR = ROOT / "tests"


def print_separator(char="=", length=80):
    print(char * length)


def get_python_files(directory):
    if not directory.exists():
        return []

    return sorted(
        [
            path
            for path in directory.rglob("*.py")
            if path.is_file()
            and "__pycache__" not in path.parts
        ],
        key=lambda path: str(path).lower(),
    )


def print_file_list(title, directory):
    print()
    print_separator()
    print(title)
    print_separator()

    if not directory.exists():
        print(f"[NOT FOUND] {directory}")
        return []

    files = get_python_files(directory)

    if not files:
        print("[NO PYTHON FILES FOUND]")
        return []

    for path in files:
        relative_path = path.relative_to(ROOT)
        size = path.stat().st_size
        print(f"{relative_path}  |  {size:,} bytes")

    return files


def print_directory_tree(directory, title):
    print()
    print_separator()
    print(title)
    print_separator()

    if not directory.exists():
        print(f"[NOT FOUND] {directory}")
        return

    all_items = sorted(
        [
            path
            for path in directory.rglob("*")
            if "__pycache__" not in path.parts
        ],
        key=lambda path: (
            len(path.relative_to(directory).parts),
            str(path).lower(),
        ),
    )

    print(directory.name + "/")

    for path in all_items:
        relative = path.relative_to(directory)
        depth = len(relative.parts) - 1
        indent = "    " * depth

        if path.is_dir():
            print(f"{indent}├── {path.name}/")
        elif path.is_file():
            print(f"{indent}├── {path.name}")


def print_statistics(src_files, test_files):
    print()
    print_separator()
    print("PROJECT STATISTICS")
    print_separator()

    print(f"Project root : {ROOT}")
    print(f"SRC path     : {SRC_DIR}")
    print(f"TESTS path   : {TESTS_DIR}")
    print()

    print(f"Python files in src   : {len(src_files)}")
    print(f"Python files in tests : {len(test_files)}")
    print(f"Total Python files    : {len(src_files) + len(test_files)}")

    src_size = sum(path.stat().st_size for path in src_files)
    tests_size = sum(path.stat().st_size for path in test_files)

    print()
    print(f"Total src size        : {src_size:,} bytes")
    print(f"Total tests size      : {tests_size:,} bytes")
    print(f"Total project size    : {src_size + tests_size:,} bytes")


def print_src_modules(src_files):
    print()
    print_separator()
    print("SRC MODULES")
    print_separator()

    if not src_files:
        print("[NO SRC MODULES FOUND]")
        return

    for path in src_files:
        relative = path.relative_to(SRC_DIR)
        print(f"- {relative}")


def print_test_modules(test_files):
    print()
    print_separator()
    print("TEST MODULES")
    print_separator()

    if not test_files:
        print("[NO TEST MODULES FOUND]")
        return

    for path in test_files:
        relative = path.relative_to(TESTS_DIR)
        print(f"- {relative}")


def print_special_files():
    print()
    print_separator()
    print("SPECIAL PROJECT FILES")
    print_separator()

    special_names = {
        "requirements.txt",
        "pyproject.toml",
        "pytest.ini",
        "setup.cfg",
        "setup.py",
        ".env",
        ".gitignore",
        "README.md",
    }

    found = []

    for name in sorted(special_names):
        path = ROOT / name

        if path.exists() and path.is_file():
            found.append(path)
            print(f"- {name}  |  {path.stat().st_size:,} bytes")

    if not found:
        print("[NONE FOUND]")


def main():
    print_separator("=")
    print("QUANTAI PROJECT STRUCTURE ANALYZER")
    print_separator("=")

    print()
    print(f"Project root: {ROOT}")

    src_files = get_python_files(SRC_DIR)
    test_files = get_python_files(TESTS_DIR)

    print_directory_tree(
        SRC_DIR,
        "[SRC] DIRECTORY TREE",
    )

    print_directory_tree(
        TESTS_DIR,
        "[TESTS] DIRECTORY TREE",
    )

    print_file_list(
        "[SRC] PYTHON FILES WITH SIZES",
        SRC_DIR,
    )

    print_file_list(
        "[TESTS] PYTHON FILES WITH SIZES",
        TESTS_DIR,
    )

    print_src_modules(src_files)

    print_test_modules(test_files)

    print_special_files()

    print_statistics(
        src_files,
        test_files,
    )

    print()
    print_separator()
    print("END OF STRUCTURE ANALYSIS")
    print_separator()


if __name__ == "__main__":
    main()