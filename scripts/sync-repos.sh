#!/bin/bash

# Tree-Mendous Repository Sync Script
# Syncs examples and docs to separate repositories using git subrepo

set -e

echo "🔄 Syncing Tree-Mendous repositories..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EXAMPLES_REPO="git@github.com:josephjohncox/tree-mendous-examples.git"
DOCS_REPO="git@github.com:josephjohncox/tree-mendous-docs.git"

# Function to check if we're in the right directory
check_directory() {
    if [[ ! -d "treemendous" ]]; then
        echo -e "${RED}❌ Error: Must be run from Tree-Mendous root directory${NC}"
        exit 1
    fi
}

# Function to check if git subrepo is installed
check_subrepo() {
    if ! command -v git-subrepo >/dev/null 2>&1; then
        echo -e "${RED}❌ Error: git-subrepo is not installed${NC}"
        echo -e "${YELLOW}Install with:${NC}"
        echo "  # macOS with Homebrew:"
        echo "  brew install git-subrepo"
        echo ""
        echo "  # Or clone and install manually:"
        echo "  git clone https://github.com/ingydotnet/git-subrepo.git"
        echo "  cd git-subrepo && make install"
        exit 1
    fi
}

# Function to initialize subrepo if needed
init_subrepo() {
    local dir=$1
    local repo=$2
    local name=$3
    
    if [[ ! -f "$dir/.gitrepo" ]]; then
        echo -e "${BLUE}  Initializing $name as subrepo...${NC}"
        if [[ -d "$dir" ]]; then
            # Directory exists, convert to subrepo
            git subrepo init "$dir"
        else
            # Directory doesn't exist, clone as subrepo
            git subrepo clone "$repo" "$dir"
        fi
    fi
}

# Function to sync examples
sync_examples() {
    echo -e "${BLUE}📊 Syncing examples to separate repository...${NC}"
    
    if [[ -d "examples" ]]; then
        init_subrepo "examples" "$EXAMPLES_REPO" "examples"
        git subrepo push examples
    else
        echo -e "${YELLOW}⚠️  Examples directory not found, cloning from remote...${NC}"
        git subrepo clone "$EXAMPLES_REPO" examples
    fi
    
    echo -e "${GREEN}✅ Examples synced successfully${NC}"
}

# Function to sync docs
sync_docs() {
    echo -e "${BLUE}📚 Syncing docs to separate repository...${NC}"
    
    if [[ -d "docs" ]]; then
        init_subrepo "docs" "$DOCS_REPO" "docs"
        git subrepo push docs
    else
        echo -e "${YELLOW}⚠️  Docs directory not found, cloning from remote...${NC}"
        git subrepo clone "$DOCS_REPO" docs
    fi
    
    echo -e "${GREEN}✅ Docs synced successfully${NC}"
}

# Function to pull changes from separate repos
pull_examples() {
    echo -e "${BLUE}📥 Pulling examples from separate repository...${NC}"
    
    if [[ -d "examples" && -f "examples/.gitrepo" ]]; then
        git subrepo pull examples
    else
        echo -e "${YELLOW}⚠️  Examples not setup as subrepo, cloning...${NC}"
        git subrepo clone "$EXAMPLES_REPO" examples
    fi
    
    echo -e "${GREEN}✅ Examples pulled successfully${NC}"
}

pull_docs() {
    echo -e "${BLUE}📥 Pulling docs from separate repository...${NC}"
    
    if [[ -d "docs" && -f "docs/.gitrepo" ]]; then
        git subrepo pull docs
    else
        echo -e "${YELLOW}⚠️  Docs not setup as subrepo, cloning...${NC}"
        git subrepo clone "$DOCS_REPO" docs
    fi
    
    echo -e "${GREEN}✅ Docs pulled successfully${NC}"
}

# Main execution
check_directory
check_subrepo

case "${1:-sync}" in
    "sync")
        sync_examples
        sync_docs
        ;;
    "examples")
        sync_examples
        ;;
    "docs")
        sync_docs
        ;;
    "pull-examples")
        pull_examples
        ;;
    "pull-docs")
        pull_docs
        ;;
    "pull")
        pull_examples
        pull_docs
        ;;
    "status")
        echo -e "${BLUE}📊 Subrepo Status:${NC}"
        if [[ -d "examples" ]]; then
            echo -e "${BLUE}Examples:${NC}"
            git subrepo status examples || echo "  Not a subrepo"
        fi
        if [[ -d "docs" ]]; then
            echo -e "${BLUE}Docs:${NC}"
            git subrepo status docs || echo "  Not a subrepo"
        fi
        ;;
    "clean")
        echo -e "${BLUE}🧹 Cleaning subrepo metadata...${NC}"
        git subrepo clean examples 2>/dev/null || true
        git subrepo clean docs 2>/dev/null || true
        echo -e "${GREEN}✅ Cleaned${NC}"
        ;;
    *)
        echo "Usage: $0 [sync|examples|docs|pull|pull-examples|pull-docs|status|clean]"
        echo ""
        echo "Commands:"
        echo "  sync          - Push examples and docs to separate repos (default)"
        echo "  examples      - Push only examples"
        echo "  docs          - Push only docs"
        echo "  pull          - Pull changes from both separate repos"
        echo "  pull-examples - Pull only examples"
        echo "  pull-docs     - Pull only docs"
        echo "  status        - Show subrepo status"
        echo "  clean         - Clean subrepo metadata"
        echo ""
        echo "Note: Requires git-subrepo to be installed:"
        echo "  brew install git-subrepo"
        exit 1
        ;;
esac

echo -e "${GREEN}🎉 Repository sync completed!${NC}"
