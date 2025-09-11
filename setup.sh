#!/bin/bash
# Setup script for Synthetic Finance ML project

# 1. Create virtual environment
python -m venv venv

# 2. Activate environment 
source venv/bin/activate  # On Windows: venv\Scripts\activate
# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

echo "✅ Setup complete! Activate the environment with 'source venv/bin/activate' (Linux/Mac) or 'venv\\Scripts\\activate' (Windows)"
