#!/bin/bash
# Della Git Workflow for Trajectory Analysis
# This script helps manage the della → git → local workflow

echo "🚀 Della Trajectory Analysis Git Workflow"
echo "=========================================="

# Function to run on della
run_on_della() {
    echo "📊 Running trajectory analysis on della..."
    
    # Run the analysis
    python della_trajectory_analysis.py \
        --search_paths /scratch/network/mg7411 /tmp /home/mg7411 . \
        --output_dir della_analysis_results
    
    echo "✅ Analysis complete!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Review results in della_analysis_results/"
    echo "2. Run: ./della_git_workflow.sh push"
}

# Function to push results to git
push_to_git() {
    echo "📤 Pushing analysis results to git..."
    
    if [ ! -d "della_analysis_results" ]; then
        echo "❌ della_analysis_results directory not found!"
        echo "   Run analysis first: ./della_git_workflow.sh analyze"
        exit 1
    fi
    
    # Add trajectory analysis results
    git add della_analysis_results/
    git add della_trajectory_analysis.py
    git add della_git_workflow.sh
    
    # Create commit with timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    git commit -m "Add della trajectory analysis results - $timestamp

- Analyzed F8D3/F16D3 trajectory data from timed-out experiments
- K=1 vs K=10 adaptation steps comparison
- Sample efficiency analysis across multiple seeds
- Camera-ready insights for ICML submission"
    
    # Push to remote
    git push origin master
    
    echo "✅ Results pushed to git!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Go to local machine"
    echo "2. Run: git pull"
    echo "3. Run: ./della_git_workflow.sh analyze_local"
}

# Function to analyze locally after pulling
analyze_local() {
    echo "💻 Setting up local analysis..."
    
    if [ ! -d "della_analysis_results" ]; then
        echo "❌ della_analysis_results directory not found!"
        echo "   Make sure you've pulled from git: git pull"
        exit 1
    fi
    
    echo "✅ Found della analysis results!"
    echo ""
    echo "📊 Available files:"
    ls -la della_analysis_results/
    echo ""
    echo "📋 Recommended actions:"
    echo "1. Review: della_analysis_results/DELLA_TRAJECTORY_ANALYSIS.md"
    echo "2. View: della_analysis_results/della_trajectory_analysis.png"
    echo "3. Create publication figures using local analysis tools"
    echo "4. Integrate with loss landscape analysis"
    echo ""
    echo "🎨 To create publication-quality figures:"
    echo "   python enhance_della_analysis_for_publication.py"
}

# Function to show status
show_status() {
    echo "📊 Current Status"
    echo "=================="
    echo ""
    
    if [ -d "della_analysis_results" ]; then
        echo "✅ Analysis results found:"
        echo "   📁 della_analysis_results/"
        if [ -f "della_analysis_results/DELLA_TRAJECTORY_ANALYSIS.md" ]; then
            echo "   📄 Analysis report available"
        fi
        if [ -f "della_analysis_results/della_trajectory_analysis.png" ]; then
            echo "   🎨 Visualization available"
        fi
        echo ""
        echo "📈 Summary from analysis:"
        if [ -f "della_analysis_results/DELLA_TRAJECTORY_ANALYSIS.md" ]; then
            echo "$(head -20 della_analysis_results/DELLA_TRAJECTORY_ANALYSIS.md)"
        fi
    else
        echo "❌ No analysis results found"
        echo "   Run analysis first if on della: ./della_git_workflow.sh analyze"
        echo "   Or pull from git if local: git pull"
    fi
}

# Function to clean up
cleanup() {
    echo "🧹 Cleaning up analysis results..."
    read -p "Are you sure you want to delete della_analysis_results/? (y/N): " confirm
    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        rm -rf della_analysis_results/
        echo "✅ Cleanup complete!"
    else
        echo "❌ Cleanup cancelled"
    fi
}

# Main script logic
case "$1" in
    "analyze"|"analyse")
        run_on_della
        ;;
    "push")
        push_to_git
        ;;
    "pull"|"local")
        analyze_local
        ;;
    "status")
        show_status
        ;;
    "clean"|"cleanup")
        cleanup
        ;;
    *)
        echo "🔧 Usage: $0 {analyze|push|local|status|clean}"
        echo ""
        echo "Commands:"
        echo "  analyze  - Run trajectory analysis on della"
        echo "  push     - Push analysis results to git"
        echo "  local    - Set up local analysis after git pull"
        echo "  status   - Show current status"
        echo "  clean    - Clean up analysis results"
        echo ""
        echo "📋 Typical workflow:"
        echo "  On della:  $0 analyze  → $0 push"
        echo "  Locally:   git pull    → $0 local"
        exit 1
        ;;
esac 