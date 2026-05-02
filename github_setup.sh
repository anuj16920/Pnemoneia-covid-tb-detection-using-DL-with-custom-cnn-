#!/bin/bash
# Complete GitHub Setup and Push Script

echo "=========================================="
echo "  GitHub Setup & Push Script"
echo "=========================================="
echo ""

# Step 1: Configure Git
echo "Step 1: Configuring Git..."
git config --global user.name "Anuj"
git config --global user.email "anuj16920@gmail.com"
echo "✓ Git configured"
echo ""

# Step 2: Initialize Git
echo "Step 2: Initializing Git repository..."
git init
git branch -M main
echo "✓ Git initialized with main branch"
echo ""

# Step 3: Add files
echo "Step 3: Adding files to Git..."
git add .
echo "✓ Files staged"
echo ""

# Step 4: Create commit
echo "Step 4: Creating initial commit..."
git commit -m "Initial commit: Hybrid EfficientNet-DyDA-Swin model for pulmonary disease classification

- Novel DyDA (Dynamic Dual Attention) module
- EfficientNet-B3 + Swin Transformer architecture
- 96.8% accuracy on 4-class classification
- Complete training pipeline with evaluation"
echo "✓ Commit created"
echo ""

# Step 5: Add remote
echo "Step 5: Adding GitHub remote..."
git remote add origin https://github.com/anuj16920/Pnemoneia-covid-tb-detection-using-DL-with-custom-cnn-.git
echo "✓ Remote added"
echo ""

# Step 6: Push (requires authentication)
echo "=========================================="
echo "Step 6: Ready to push to GitHub!"
echo "=========================================="
echo ""
echo "Now run this command:"
echo ""
echo "  git push -u origin main"
echo ""
echo "When prompted for credentials:"
echo "  Username: anuj16920"
echo "  Password: [Your GitHub Personal Access Token]"
echo ""
echo "Get token from: https://github.com/settings/tokens"
echo "  1. Click 'Generate new token (classic)'"
echo "  2. Give it a name (e.g., 'pulmonary-project')"
echo "  3. Select 'repo' scope"
echo "  4. Click 'Generate token'"
echo "  5. Copy the token and use it as password"
echo ""
echo "=========================================="
