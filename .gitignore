# Python cache files and build artifacts
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Virtual environment (common names - uncomment if used within project)
# venv/
# env/
# .env/
# */venv/
# */env/
# */.env/

# IDE / Editor specific files (optional but recommended)
.vscode/
.idea/
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?
*~

# OS specific files (optional but recommended)
.DS_Store
Thumbs.db

# Log files
*.log
logs/ # Ignore top-level logs directory if you create one

# Temporary Files created by scripts (if any)
temp/

# Jupyter Notebook Checkpoints (if applicable)
.ipynb_checkpoints/

# ==================================================
# --- Project Specific Large File/Directory Ignores ---
# ==================================================

# 1. Ignore the entire main data directory
# This contains raw, processed, generated, and organized images.
data/

# 2. Ignore the entire experiments directory
# This contains checkpoints, generated samples, evaluation outputs, tensorboard logs.
expr/

# 3. Ignore the directory for potentially large local models/weights
my_models/

# 4. Ignore specific large weight/checkpoint files if not already covered
#    (Usually covered by ignoring 'expr/' and 'my_models/')
# *.pth
# *.ckpt
# *.h5
# *.npy

# --- Optional: Exclude LPIPS weights file ---
# Uncomment the next line if you don't want to commit the LPIPS weights
# metrics/lpips_weights.ckpt

# --- Note on what IS included ---
# This .gitignore allows committing:
# - All Python scripts (.py) in the root and subdirectories (models, datasets, etc.)
# - Configuration files (configs/*.yaml)
# - requirements.txt
# - README.md
# - The .gitignore file itself
# - Necessary files within metrics/ (like fid.py, lpips.py) unless excluded above
# - (Potentially) K-Fold CSVs if you explicitly un-ignore them later with !data/...