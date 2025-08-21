# install_dependencies.py
"""
This script installs all required Python packages for the project.
Run this file in your environment before executing any notebooks.
"""
import subprocess
import sys

required_packages = [
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'scikit-learn',
    'requests'
]

for package in required_packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

print('All required packages installed successfully.')
