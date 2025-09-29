// High-Performance Treap Implementation for Interval Trees
// Combines binary search tree with randomized heap priorities
// Provides O(log n) expected performance with high probability

#include <random>
#include <vector>
#include <optional>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>

class TreapNode {
public:
    int start, end;
    double priority;
    int height;
    int subtree_size;
    int total_length;
    
    TreapNode* left;
    TreapNode* right;
    
    TreapNode(int s, int e, double p = -1.0) 
        : start(s), end(e), height(1), subtree_size(1), left(nullptr), right(nullptr) {
        total_length = end - start;
        
        // Generate random priority if not provided
        if (p < 0) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<double> dis(0.0, 1.0);
            priority = dis(gen);
        } else {
            priority = p;
        }
    }
    
    ~TreapNode() {
        delete left;
        delete right;
    }
    
    void update_stats() {
        // Update height
        height = 1 + std::max(get_height(left), get_height(right));
        
        // Update total length
        total_length = end - start;
        if (left) total_length += left->total_length;
        if (right) total_length += right->total_length;
        
        // Update subtree size
        subtree_size = 1;
        if (left) subtree_size += left->subtree_size;
        if (right) subtree_size += right->subtree_size;
    }
    
    static int get_height(TreapNode* node) {
        return node ? node->height : 0;
    }
    
    static int get_size(TreapNode* node) {
        return node ? node->subtree_size : 0;
    }
    
    int length() const {
        return end - start;
    }
};

class IntervalTreap {
public:
    IntervalTreap() : root(nullptr), rng(std::random_device{}()) {}
    
    explicit IntervalTreap(unsigned int seed) : root(nullptr), rng(seed) {}
    
    ~IntervalTreap() {
        delete root;
    }
    
    // Disable copy constructor and assignment for simplicity
    IntervalTreap(const IntervalTreap&) = delete;
    IntervalTreap& operator=(const IntervalTreap&) = delete;
    
    // Move constructor and assignment
    IntervalTreap(IntervalTreap&& other) noexcept : root(other.root), rng(std::move(other.rng)) {
        other.root = nullptr;
    }
    
    IntervalTreap& operator=(IntervalTreap&& other) noexcept {
        if (this != &other) {
            delete root;
            root = other.root;
            other.root = nullptr;
            rng = std::move(other.rng);
        }
        return *this;
    }
    
    void release_interval(int start, int end) {
        if (start >= end) return;
        
        // Remove overlapping intervals and merge
        auto overlapping = find_and_remove_overlapping(start, end);
        
        // Merge with overlapping intervals
        int merged_start = start;
        int merged_end = end;
        
        for (const auto& interval : overlapping) {
            merged_start = std::min(merged_start, interval.first);
            merged_end = std::max(merged_end, interval.second);
        }
        
        // Insert merged interval
        insert_interval(merged_start, merged_end);
    }
    
    void reserve_interval(int start, int end) {
        if (start >= end) return;
        root = delete_range(root, start, end);
    }
    
    std::optional<std::pair<int, int>> find_interval(int start, int length) {
        TreapNode* result = find_interval_node(root, start, length);
        if (result) {
            // Allocate at the requested start point if possible, otherwise at interval start
            int alloc_start = std::max(start, result->start);
            if (result->end - alloc_start >= length) {
                return std::make_pair(alloc_start, alloc_start + length);
            }
        }
        return std::nullopt;
    }
    
    std::vector<std::pair<int, int>> get_intervals() const {
        std::vector<std::pair<int, int>> result;
        inorder_traversal(root, result);
        return result;
    }
    
    int get_total_available_length() const {
        return root ? root->total_length : 0;
    }
    
    int get_tree_size() const {
        return root ? root->subtree_size : 0;
    }
    
    int get_height() const {
        return root ? root->height : 0;
    }
    
    double get_expected_height() const {
        int size = get_tree_size();
        return size > 0 ? std::log2(size + 1) : 0.0;
    }
    
    // Treap-specific operations
    std::optional<std::pair<int, int>> sample_random_interval() {
        if (!root) return std::nullopt;
        
        std::uniform_int_distribution<int> dis(0, root->subtree_size - 1);
        int target_index = dis(rng);
        
        TreapNode* node = select_kth_interval(root, target_index);
        if (node) {
            return std::make_pair(node->start, node->end);
        }
        return std::nullopt;
    }
    
    std::pair<IntervalTreap, IntervalTreap> split(int key) {
        IntervalTreap left_treap, right_treap;
        auto [left_root, right_root] = split_at_key(root, key);
        
        left_treap.root = left_root;
        right_treap.root = right_root;
        
        // Prevent double deletion
        root = nullptr;
        
        return {std::move(left_treap), std::move(right_treap)};
    }
    
    std::vector<std::pair<int, int>> find_overlapping_intervals(int start, int end) const {
        std::vector<std::pair<int, int>> result;
        find_overlapping(root, start, end, result);
        return result;
    }
    
    bool verify_treap_properties() const {
        return verify_bst_property(root) && verify_heap_property(root);
    }
    
    struct TreapStatistics {
        int size;
        int height;
        double expected_height;
        double balance_factor;
        int total_length;
        double avg_interval_length;
    };
    
    TreapStatistics get_statistics() const {
        TreapStatistics stats{};
        
        if (!root) {
            return stats;
        }
        
        stats.size = root->subtree_size;
        stats.height = root->height;
        stats.expected_height = std::log2(stats.size + 1);
        stats.balance_factor = stats.expected_height > 0 ? 
                              static_cast<double>(stats.height) / stats.expected_height : 1.0;
        stats.total_length = root->total_length;
        stats.avg_interval_length = static_cast<double>(stats.total_length) / stats.size;
        
        return stats;
    }
    
    void print_tree() const {
        print_tree_helper(root, "", "");
    }

private:
    TreapNode* root;
    mutable std::mt19937 rng;
    
    void insert_interval(int start, int end) {
        double priority = generate_random_priority();
        TreapNode* new_node = new TreapNode(start, end, priority);
        root = insert(root, new_node);
    }
    
    double generate_random_priority() {
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        return dis(rng);
    }
    
    TreapNode* insert(TreapNode* node, TreapNode* new_node) {
        if (!node) {
            return new_node;
        }
        
        // BST insertion by start time
        if (new_node->start < node->start) {
            node->left = insert(node->left, new_node);
            // Rotate right if heap property violated
            if (node->left && node->left->priority > node->priority) {
                node = rotate_right(node);
            }
        } else {
            node->right = insert(node->right, new_node);
            // Rotate left if heap property violated  
            if (node->right && node->right->priority > node->priority) {
                node = rotate_left(node);
            }
        }
        
        node->update_stats();
        return node;
    }
    
    TreapNode* delete_range(TreapNode* node, int start, int end) {
        if (!node) return nullptr;
        
        // Check for overlap
        if (node->end <= start) {
            // No overlap, search right
            node->right = delete_range(node->right, start, end);
        } else if (node->start >= end) {
            // No overlap, search left
            node->left = delete_range(node->left, start, end);
        } else {
            // Overlap - need to handle this node
            std::vector<TreapNode*> remainders;
            
            // Create left remainder if needed
            if (node->start < start) {
                // Use priority lower than original node to maintain heap property
                remainders.push_back(new TreapNode(node->start, start, node->priority * 0.5));
            }
            
            // Create right remainder if needed
            if (node->end > end) {
                // Use priority lower than original node to maintain heap property
                remainders.push_back(new TreapNode(end, node->end, node->priority * 0.5));
            }
            
            // Delete current node and process subtrees
            TreapNode* left_subtree = delete_range(node->left, start, end);
            TreapNode* right_subtree = delete_range(node->right, start, end);
            
            // Avoid double deletion
            node->left = nullptr;
            node->right = nullptr;
            delete node;
            
            node = merge_subtrees(left_subtree, right_subtree);
            
            // Insert remainders
            for (TreapNode* remainder : remainders) {
                node = insert(node, remainder);
            }
        }
        
        if (node) {
            node->update_stats();
        }
        return node;
    }
    
    std::vector<std::pair<int, int>> find_and_remove_overlapping(int start, int end) {
        std::vector<std::pair<int, int>> overlapping;
        root = collect_and_remove_overlapping(root, start, end, overlapping);
        return overlapping;
    }
    
    TreapNode* collect_and_remove_overlapping(TreapNode* node, int start, int end,
                                            std::vector<std::pair<int, int>>& overlapping) {
        if (!node) return nullptr;
        
        if (node->end <= start) {
            node->right = collect_and_remove_overlapping(node->right, start, end, overlapping);
        } else if (node->start >= end) {
            node->left = collect_and_remove_overlapping(node->left, start, end, overlapping);
        } else {
            // Overlap found
            overlapping.emplace_back(node->start, node->end);
            
            // Remove this node and continue in subtrees
            TreapNode* left_subtree = collect_and_remove_overlapping(node->left, start, end, overlapping);
            TreapNode* right_subtree = collect_and_remove_overlapping(node->right, start, end, overlapping);
            
            // Avoid double deletion
            node->left = nullptr;
            node->right = nullptr;
            delete node;
            
            return merge_subtrees(left_subtree, right_subtree);
        }
        
        if (node) {
            node->update_stats();
        }
        return node;
    }
    
    TreapNode* find_interval_node(TreapNode* node, int start, int length) {
        if (!node) return nullptr;
        
        // Simple implementation: find any interval that can accommodate the request
        // Check if current node works
        if (node->start <= start && start < node->end && (node->end - start) >= length) {
            return node;  // Found suitable interval
        }
        
        // If start is before this node, try left subtree first
        if (start < node->start) {
            TreapNode* left_result = find_interval_node(node->left, start, length);
            if (left_result) return left_result;
        }
        
        // Try right subtree
        return find_interval_node(node->right, start, length);
    }
    
    TreapNode* merge_subtrees(TreapNode* left, TreapNode* right) {
        if (!left) return right;
        if (!right) return left;
        
        // Choose root based on priority (heap property)
        if (left->priority > right->priority) {
            left->right = merge_subtrees(left->right, right);
            left->update_stats();
            return left;
        } else {
            right->left = merge_subtrees(left, right->left);
            right->update_stats();
            return right;
        }
    }
    
    TreapNode* rotate_left(TreapNode* node) {
        if (!node || !node->right) return node;
        
        TreapNode* new_root = node->right;
        node->right = new_root->left;
        new_root->left = node;
        
        node->update_stats();
        new_root->update_stats();
        return new_root;
    }
    
    TreapNode* rotate_right(TreapNode* node) {
        if (!node || !node->left) return node;
        
        TreapNode* new_root = node->left;
        node->left = new_root->right;
        new_root->right = node;
        
        node->update_stats();
        new_root->update_stats();
        return new_root;
    }
    
    std::pair<TreapNode*, TreapNode*> split_at_key(TreapNode* node, int key) {
        if (!node) {
            return {nullptr, nullptr};
        }
        
        if (node->start < key) {
            auto [left, right] = split_at_key(node->right, key);
            node->right = left;
            node->update_stats();
            return {node, right};
        } else {
            auto [left, right] = split_at_key(node->left, key);
            node->left = right;
            node->update_stats();
            return {left, node};
        }
    }
    
    TreapNode* select_kth_interval(TreapNode* node, int k) {
        if (!node) return nullptr;
        
        int left_size = TreapNode::get_size(node->left);
        
        if (k < left_size) {
            return select_kth_interval(node->left, k);
        } else if (k == left_size) {
            return node;
        } else {
            return select_kth_interval(node->right, k - left_size - 1);
        }
    }
    
    void inorder_traversal(TreapNode* node, std::vector<std::pair<int, int>>& result) const {
        if (!node) return;
        
        inorder_traversal(node->left, result);
        result.emplace_back(node->start, node->end);
        inorder_traversal(node->right, result);
    }
    
    void find_overlapping(TreapNode* node, int start, int end, 
                         std::vector<std::pair<int, int>>& result) const {
        if (!node) return;
        
        // Check overlap with current node
        if (node->start < end && node->end > start) {
            result.emplace_back(node->start, node->end);
        }
        
        // Search subtrees
        find_overlapping(node->left, start, end, result);
        find_overlapping(node->right, start, end, result);
    }
    
    bool verify_bst_property(TreapNode* node) const {
        if (!node) return true;
        
        if (node->left && node->left->start >= node->start) return false;
        if (node->right && node->right->start < node->start) return false;
        
        return verify_bst_property(node->left) && verify_bst_property(node->right);
    }
    
    bool verify_heap_property(TreapNode* node) const {
        if (!node) return true;
        
        if (node->left && node->left->priority > node->priority) return false;
        if (node->right && node->right->priority > node->priority) return false;
        
        return verify_heap_property(node->left) && verify_heap_property(node->right);
    }
    
    void print_tree_helper(TreapNode* node, const std::string& indent, const std::string& prefix) const {
        if (!node) return;
        
        print_tree_helper(node->right, indent + "    ", "┌── ");
        std::cout << indent << prefix << "[" << node->start << "," << node->end 
                  << ") p=" << std::fixed << std::setprecision(3) << node->priority
                  << " (h=" << node->height << ", s=" << node->subtree_size << ")" << std::endl;
        print_tree_helper(node->left, indent + "    ", "└── ");
    }
};

// Performance-optimized treap with additional features
class HighPerformanceTreap : public IntervalTreap {
public:
    HighPerformanceTreap() : IntervalTreap() {
        // Pre-allocate node pool for better performance
        node_pool.reserve(1000);
    }
    
    // Bulk operations for improved performance
    void bulk_insert(const std::vector<std::pair<int, int>>& intervals) {
        // Sort intervals and insert in batches for better cache performance
        auto sorted_intervals = intervals;
        std::sort(sorted_intervals.begin(), sorted_intervals.end());
        
        for (const auto& [start, end] : sorted_intervals) {
            IntervalTreap::release_interval(start, end);
        }
    }
    
    void bulk_delete(const std::vector<std::pair<int, int>>& intervals) {
        // Process deletions in batches
        for (const auto& [start, end] : intervals) {
            IntervalTreap::reserve_interval(start, end);
        }
    }
    
    // Memory pool for node allocation (simplified)
    std::vector<TreapNode> node_pool;
    size_t pool_index = 0;
    
    TreapNode* allocate_node(int start, int end, double priority = -1.0) {
        if (pool_index < node_pool.size()) {
            TreapNode& node = node_pool[pool_index++];
            node = TreapNode(start, end, priority);
            return &node;
        } else {
            return new TreapNode(start, end, priority);
        }
    }
};

// Example usage
#ifdef TREAP_STANDALONE_TEST
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "🌳 C++ Treap Interval Tree Test" << std::endl;
    std::cout << "===============================" << std::endl;
    
    IntervalTreap treap(42);  // Fixed seed for reproducible results
    
    // Insert test intervals
    std::vector<std::pair<int, int>> test_intervals = {
        {10, 20}, {30, 40}, {15, 25}, {50, 60}, {5, 15}
    };
    
    std::cout << "Inserting intervals..." << std::endl;
    for (const auto& [start, end] : test_intervals) {
        treap.release_interval(start, end);
        std::cout << "  Inserted [" << start << ", " << end << ")" << std::endl;
    }
    
    // Display statistics
    auto stats = treap.get_statistics();
    std::cout << "\nTreap Statistics:" << std::endl;
    std::cout << "  Size: " << stats.size << std::endl;
    std::cout << "  Height: " << stats.height << " (expected: " 
              << std::fixed << std::setprecision(1) << stats.expected_height << ")" << std::endl;
    std::cout << "  Balance factor: " << std::setprecision(2) << stats.balance_factor << std::endl;
    std::cout << "  Total length: " << stats.total_length << std::endl;
    
    // Verify properties
    std::cout << "\nProperty verification: " 
              << (treap.verify_treap_properties() ? "✅ PASS" : "❌ FAIL") << std::endl;
    
    // Test operations
    std::cout << "\nTesting operations:" << std::endl;
    
    auto overlapping = treap.find_overlapping_intervals(12, 35);
    std::cout << "  Overlapping with [12, 35): ";
    for (const auto& [start, end] : overlapping) {
        std::cout << "[" << start << "," << end << ") ";
    }
    std::cout << std::endl;
    
    // Random sampling
    std::cout << "  Random samples: ";
    for (int i = 0; i < 3; ++i) {
        auto sample = treap.sample_random_interval();
        if (sample) {
            std::cout << "[" << sample->first << "," << sample->second << ") ";
        }
    }
    std::cout << std::endl;
    
    std::cout << "\n✅ C++ Treap test complete!" << std::endl;
    
    return 0;
}
#endif
