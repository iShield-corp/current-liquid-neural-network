#!/bin/bash
# Test script to verify git tracking of essential files

echo "🔍 Checking if essential Python files are tracked by git..."

# Check if key Python files are tracked
key_files=(
    "src/__init__.py"
    "src/core/__init__.py"
    "src/datasets/__init__.py"
    "src/training/__init__.py"
    "tests/__init__.py"
    "scripts/cli.py"
    "src/core/main.py"
    "src/datasets/advanced_programming_datasets.py"
    "src/datasets/vision_datasets.py"
    "src/datasets/robotics_datasets.py"
)

missing_files=()

for file in "${key_files[@]}"; do
    if git ls-files --error-unmatch "$file" > /dev/null 2>&1; then
        echo "✅ $file is tracked"
    else
        echo "❌ $file is NOT tracked"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo ""
    echo "🎉 All essential files are properly tracked by git!"
else
    echo ""
    echo "⚠️  Missing files that need to be added:"
    for file in "${missing_files[@]}"; do
        echo "   git add $file"
    done
    
    echo ""
    echo "Run the following commands to fix:"
    echo "git add ."
    echo "git commit -m 'Add missing Python package files'"
    echo "git push"
fi

echo ""
echo "📋 Current git status:"
git status --porcelain
