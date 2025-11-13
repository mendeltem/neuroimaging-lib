# Installation Guide for neuroimaging-lib

## Directory Structure

Your package should look like this:

```
neuroimaging-lib/
├── neuroimaging_lib/          # ← Package folder (IMPORTANT: use underscore)
│   ├── __init__.py            # Makes it a package & exports functions
│   └── directory_lib.py       # Your optimized functions
├── tests/                      # (Optional) Test files
├── setup.py                    # Installation script
├── pyproject.toml             # Modern build configuration
├── MANIFEST.in                # Include extra files in distribution
├── requirements.txt           # Dependencies list
├── README.md                  # Documentation
└── LICENSE                    # License file (optional but recommended)
```

## Installation Methods

### Method 1: Development Installation (Recommended for Development)

Best when you're actively working on the code:

```bash
cd /path/to/neuroimaging-lib
pip install -e .
```

**What this does:**
- Installs the package in "editable" mode
- Changes to your code are immediately reflected
- No need to reinstall after each change

### Method 2: Regular Installation

For when you want to freeze a version:

```bash
cd /path/to/neuroimaging-lib
pip install .
```

### Method 3: Install with Requirements First

To ensure all dependencies are properly installed:

```bash
cd /path/to/neuroimaging-lib
pip install -r requirements.txt
pip install .
```

### Method 4: Install with Development Tools

For development, testing, and code quality tools:

```bash
cd /path/to/neuroimaging-lib
pip install -e ".[dev]"
```

This installs:
- pytest (unit testing)
- pytest-cov (coverage reports)
- black (code formatting)
- flake8 (linting)
- mypy (type checking)

## Verification

After installation, verify it works:

```bash
python -c "from neuroimaging_lib import get_files_from_dir; print('✓ Installation successful!')"
```

Or in Python interactive mode:

```python
>>> from neuroimaging_lib import get_files_from_dir, get_subdirectories
>>> print(get_files_from_dir.__doc__)
```

## Usage After Installation

Once installed, you can use it in any Python project:

```python
# In any Python script or Jupyter notebook:
from neuroimaging_lib import get_files_from_dir, find_elements

# Use the functions directly
files = get_files_from_dir("/path/to/data")
```

## Troubleshooting

### Issue: "No module named 'neuroimaging_lib'"

**Solution**: Make sure you're in the correct directory and the package name matches:

```bash
# Correct
ls neuroimaging_lib/__init__.py    # Should exist

# Then install
pip install -e .
```

### Issue: "setup.py not found"

**Solution**: Make sure you're in the root directory of the package (where setup.py is):

```bash
pwd                    # Check current directory
ls setup.py           # Should exist
```

### Issue: Installation fails with dependency errors

**Solution**: Install dependencies first:

```bash
pip install nibabel>=5.0.0 nipype>=1.8.0 ants>=0.3.0
pip install -e .
```

### Issue: "Permission denied" error

**Solution**: Install for current user only:

```bash
pip install --user -e .
```

## Publishing to PyPI (Future)

When you're ready to publish to the Python Package Index:

1. Create an account at https://pypi.org
2. Install build tools:
   ```bash
   pip install build twine
   ```
3. Build the package:
   ```bash
   python -m build
   ```
4. Upload to PyPI:
   ```bash
   twine upload dist/*
   ```

Then anyone can install with:
```bash
pip install neuroimaging-lib
```

## Package Configuration Files Explained

### `setup.py`
- Traditional setuptools configuration
- Defines package metadata, dependencies, and installation behavior
- Still supported but `pyproject.toml` is preferred

### `pyproject.toml`
- Modern PEP 517/518 standard
- Contains build system requirements and project metadata
- Can be used alone without `setup.py` (if using pyproject.toml only)

### `MANIFEST.in`
- Specifies which extra files to include in distributions
- Important for including README, LICENSE, etc.

### `requirements.txt`
- Simple list of package dependencies
- Can be used with `pip install -r requirements.txt`

## Development Workflow

```bash
# 1. Clone/navigate to the project
cd /path/to/neuroimaging-lib

# 2. Install in development mode with dev tools
pip install -e ".[dev]"

# 3. Make changes to code
nano neuroimaging_lib/directory_lib.py

# 4. Test your changes
pytest

# 5. Format code
black neuroimaging_lib/

# 6. Check for type errors
mypy neuroimaging_lib/

# 7. Commit and push
git add .
git commit -m "Add new feature"
git push
```

## Key Points

✅ **Package name**: Use `neuroimaging_lib` (with underscore) as the folder name  
✅ **Import name**: Same as folder name: `from neuroimaging_lib import ...`  
✅ **Install locally**: `pip install -e .` for development  
✅ **Dependencies**: Listed in both `requirements.txt` and `setup.py`/`pyproject.toml`  
✅ **Type hints**: Already included for better IDE support  
✅ **Logging**: Uses `logging` module instead of `print()`  

## Next Steps

1. Copy the files to your project directory
2. Update `setup.py` with your actual contact info and GitHub URL
3. Run `pip install -e .`
4. Start using: `from neuroimaging_lib import get_files_from_dir`
5. (Optional) Create tests in a `tests/` directory
6. (Optional) Add `LICENSE` file (MIT recommended)

---

**Questions?** Check the README.md file for more details on API usage.