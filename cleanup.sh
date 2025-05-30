#!/bin/bash

# Remove Python cache and compiled files
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.py[co]" -delete
find . -type d -name "*.egg-info" -exec rm -r {} +

# Remove build and distribution directories
rm -rf build/ dist/ *.egg-info/

# Remove virtual environment
rm -rf venv/ env/ .venv/

# Remove IDE specific files
rm -rf .idea/ .vscode/ .project .pydevproject .settings/

# Remove log files
rm -rf logs/*.log

# Remove local development files
rm -f local_settings.py db.sqlite3 db.sqlite3-journal

# Remove temporary files
find . -type f -name "*.tmp" -delete
find . -type f -name "*.bak" -delete
find . -type f -name "*.swp" -delete
find . -type f -name "*~" -delete

# Remove Jupyter notebook checkpoints
rm -rf .ipynb_checkpoints/

echo "Cleanup complete!"
