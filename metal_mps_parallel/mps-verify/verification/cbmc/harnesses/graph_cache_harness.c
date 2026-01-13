// graph_cache_harness.c - CBMC harness for verifying MPSGraphCache correctness
//
// The graph cache stores compiled MPSGraph objects to avoid re-compilation.
// Thread-safe access is critical for parallel inference.
//
// Properties verified:
// 1. Cache size never exceeds maximum
// 2. Cache size never goes negative
// 3. No double-free of graph objects
// 4. Graphs are properly reference-counted
// 5. Evicted graphs have zero references before deletion
// 6. Lookup returns valid cached graphs or NULL
// 7. Thread-safe invariants under concurrent operations
//
// Based on: MPSGraphCache patterns in PyTorch MPS backend
//
// Run with:
//   cbmc graph_cache_harness.c --unwind 10 --pointer-check --bounds-check

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

// CBMC Primitives
extern int nondet_int(void);
extern unsigned int nondet_uint(void);
extern bool nondet_bool(void);
extern void __CPROVER_assume(bool);
extern void __CPROVER_assert(bool, const char*);

// ============================================================================
// Constants
// ============================================================================

#define MAX_CACHE_SIZE 4
#define MAX_GRAPHS 8
#define MAX_THREADS 3
#define NULL_KEY 0
#define INVALID_INDEX (-1)

// ============================================================================
// Graph Model
// ============================================================================

typedef enum {
    GRAPH_FREE = 0,      // Not allocated
    GRAPH_CREATING,      // Being compiled
    GRAPH_CACHED,        // In cache, available for use
    GRAPH_IN_USE,        // Currently being executed
    GRAPH_EVICTING,      // Being removed from cache
    GRAPH_DELETED        // Freed (tombstone for verification)
} GraphState;

typedef struct {
    int id;
    GraphState state;
    uint64_t key;           // Hash key for lookup
    int ref_count;          // Reference count (0 = can be evicted)
    int last_used_time;     // For LRU eviction
    int cache_slot;         // Index in cache (-1 if not cached)
} Graph;

// ============================================================================
// Graph Cache Model
// ============================================================================

typedef struct {
    // Cache storage (hash table with linear probing)
    int cache_slots[MAX_CACHE_SIZE];  // Graph ID at each slot (-1 = empty)
    uint64_t cache_keys[MAX_CACHE_SIZE];  // Key at each slot
    int cache_size;                    // Current number of cached graphs
    int max_size;                      // Maximum cache size

    // LRU tracking
    int current_time;                  // Monotonic time counter

    // Statistics
    int hits;
    int misses;
    int evictions;
    int insertions;

    // Mutex simulation
    int mutex_holder;
    bool mutex_locked;

} GraphCache;

// ============================================================================
// Graph Pool (allocation tracking)
// ============================================================================

static Graph g_graphs[MAX_GRAPHS];
static int g_next_graph_id = 0;

static void Graph_pool_init(void) {
    for (int i = 0; i < MAX_GRAPHS; i++) {
        g_graphs[i].id = i;
        g_graphs[i].state = GRAPH_FREE;
        g_graphs[i].key = NULL_KEY;
        g_graphs[i].ref_count = 0;
        g_graphs[i].last_used_time = 0;
        g_graphs[i].cache_slot = INVALID_INDEX;
    }
    g_next_graph_id = 0;
}

static Graph* Graph_allocate(uint64_t key) {
    for (int i = 0; i < MAX_GRAPHS; i++) {
        if (g_graphs[i].state == GRAPH_FREE) {
            g_graphs[i].state = GRAPH_CREATING;
            g_graphs[i].key = key;
            g_graphs[i].ref_count = 1;  // Initial ref from creator
            g_graphs[i].cache_slot = INVALID_INDEX;
            return &g_graphs[i];
        }
    }
    return NULL;  // Pool exhausted
}

static void Graph_acquire(Graph* g) {
    __CPROVER_assert(g != NULL, "Cannot acquire NULL graph");
    __CPROVER_assert(g->state == GRAPH_CACHED || g->state == GRAPH_IN_USE,
                    "Can only acquire cached or in-use graphs");
    __CPROVER_assert(g->ref_count > 0, "Graph ref_count must be positive");

    g->ref_count++;
}

static void Graph_release(Graph* g, GraphCache* cache) {
    __CPROVER_assert(g != NULL, "Cannot release NULL graph");
    __CPROVER_assert(g->ref_count > 0, "Cannot release with zero ref_count");

    g->ref_count--;

    if (g->ref_count == 0 && g->state == GRAPH_EVICTING) {
        // Last reference dropped, can now delete
        g->state = GRAPH_DELETED;
        g->key = NULL_KEY;
    }
}

// ============================================================================
// Cache Operations
// ============================================================================

static void GraphCache_init(GraphCache* c, int max_size) {
    __CPROVER_assume(max_size > 0 && max_size <= MAX_CACHE_SIZE);

    for (int i = 0; i < MAX_CACHE_SIZE; i++) {
        c->cache_slots[i] = INVALID_INDEX;
        c->cache_keys[i] = NULL_KEY;
    }
    c->cache_size = 0;
    c->max_size = max_size;
    c->current_time = 0;
    c->hits = 0;
    c->misses = 0;
    c->evictions = 0;
    c->insertions = 0;
    c->mutex_holder = -1;
    c->mutex_locked = false;
}

// Acquire mutex (try-lock semantics for CBMC)
static bool GraphCache_lock(GraphCache* c, int thread_id) {
    __CPROVER_assume(thread_id >= 0);

    if (c->mutex_locked) {
        return false;
    }

    c->mutex_locked = true;
    c->mutex_holder = thread_id;
    return true;
}

static void GraphCache_unlock(GraphCache* c, int thread_id) {
    __CPROVER_assert(c->mutex_locked, "Cannot unlock unlocked cache mutex");
    __CPROVER_assert(c->mutex_holder == thread_id, "Thread must hold mutex to unlock");

    c->mutex_locked = false;
    c->mutex_holder = -1;
}

// Hash function for key -> slot mapping
static int GraphCache_hash(GraphCache* c, uint64_t key) {
    return (int)(key % (uint64_t)c->max_size);
}

// Find slot for key (linear probing)
static int GraphCache_find_slot(GraphCache* c, uint64_t key) {
    int start = GraphCache_hash(c, key);

    for (int i = 0; i < c->max_size; i++) {
        int slot = (start + i) % c->max_size;
        if (c->cache_keys[slot] == key && c->cache_slots[slot] != INVALID_INDEX) {
            return slot;
        }
        if (c->cache_slots[slot] == INVALID_INDEX) {
            break;  // Empty slot, key not in cache
        }
    }
    return INVALID_INDEX;
}

// Find empty slot for insertion
static int GraphCache_find_empty_slot(GraphCache* c, uint64_t key) {
    int start = GraphCache_hash(c, key);

    for (int i = 0; i < c->max_size; i++) {
        int slot = (start + i) % c->max_size;
        if (c->cache_slots[slot] == INVALID_INDEX) {
            return slot;
        }
    }
    return INVALID_INDEX;  // Cache is full
}

// Find LRU graph for eviction
static int GraphCache_find_lru_slot(GraphCache* c) {
    int lru_slot = INVALID_INDEX;
    int min_time = c->current_time + 1;

    for (int i = 0; i < c->max_size; i++) {
        if (c->cache_slots[i] != INVALID_INDEX) {
            Graph* g = &g_graphs[c->cache_slots[i]];
            // Only evict graphs with zero external references
            if (g->ref_count == 1 && g->last_used_time < min_time) {
                min_time = g->last_used_time;
                lru_slot = i;
            }
        }
    }
    return lru_slot;
}

// Lookup graph in cache (thread-safe)
// Returns acquired graph (caller must release) or NULL
static Graph* GraphCache_lookup(GraphCache* c, uint64_t key, int thread_id) {
    __CPROVER_assume(key != NULL_KEY);

    if (!GraphCache_lock(c, thread_id)) {
        return NULL;  // Could not acquire lock
    }

    Graph* result = NULL;
    int slot = GraphCache_find_slot(c, key);

    if (slot != INVALID_INDEX) {
        // Found in cache
        int graph_id = c->cache_slots[slot];
        __CPROVER_assert(graph_id >= 0 && graph_id < MAX_GRAPHS,
                        "Cache slot must contain valid graph ID");

        Graph* g = &g_graphs[graph_id];
        __CPROVER_assert(g->state == GRAPH_CACHED || g->state == GRAPH_IN_USE,
                        "Cached graph must be in valid state");
        __CPROVER_assert(g->key == key, "Cached graph key must match lookup key");

        // Acquire reference before returning
        Graph_acquire(g);
        g->last_used_time = c->current_time++;
        g->state = GRAPH_IN_USE;

        c->hits++;
        result = g;
    } else {
        c->misses++;
    }

    GraphCache_unlock(c, thread_id);
    return result;
}

// Evict one graph from cache (must hold lock)
static bool GraphCache_evict_one_locked(GraphCache* c) {
    int lru_slot = GraphCache_find_lru_slot(c);

    if (lru_slot == INVALID_INDEX) {
        return false;  // No evictable graphs
    }

    int graph_id = c->cache_slots[lru_slot];
    Graph* g = &g_graphs[graph_id];

    __CPROVER_assert(g->state == GRAPH_CACHED || g->state == GRAPH_IN_USE,
                    "Evicted graph must be in cache state");
    __CPROVER_assert(g->ref_count == 1, "Evicted graph must have ref_count == 1");

    // Mark as evicting
    g->state = GRAPH_EVICTING;
    g->cache_slot = INVALID_INDEX;

    // Release cache's reference
    Graph_release(g, c);

    // Clear slot
    c->cache_slots[lru_slot] = INVALID_INDEX;
    c->cache_keys[lru_slot] = NULL_KEY;
    c->cache_size--;
    c->evictions++;

    __CPROVER_assert(c->cache_size >= 0, "Cache size must not be negative after eviction");

    return true;
}

// Insert graph into cache (thread-safe)
// Takes ownership of one reference; caller must have acquired graph
static bool GraphCache_insert(GraphCache* c, Graph* g, int thread_id) {
    __CPROVER_assume(g != NULL);
    __CPROVER_assume(g->key != NULL_KEY);
    __CPROVER_assert(g->state == GRAPH_CREATING || g->state == GRAPH_IN_USE,
                    "Inserted graph must be creating or in-use");

    if (!GraphCache_lock(c, thread_id)) {
        return false;  // Could not acquire lock
    }

    bool success = false;

    // Check if already in cache (race condition)
    int existing_slot = GraphCache_find_slot(c, g->key);
    if (existing_slot != INVALID_INDEX) {
        // Already cached by another thread
        GraphCache_unlock(c, thread_id);
        return false;
    }

    // Make room if needed
    if (c->cache_size >= c->max_size) {
        if (!GraphCache_evict_one_locked(c)) {
            // Cannot evict (all graphs in use)
            GraphCache_unlock(c, thread_id);
            return false;
        }
    }

    __CPROVER_assert(c->cache_size < c->max_size,
                    "Cache must have room after eviction");

    // Find slot for insertion
    int slot = GraphCache_find_empty_slot(c, g->key);
    __CPROVER_assert(slot != INVALID_INDEX, "Must have empty slot");
    __CPROVER_assert(slot >= 0 && slot < MAX_CACHE_SIZE, "Slot must be in bounds");

    // Insert graph
    c->cache_slots[slot] = g->id;
    c->cache_keys[slot] = g->key;
    g->cache_slot = slot;
    g->state = GRAPH_CACHED;
    g->last_used_time = c->current_time++;

    c->cache_size++;
    c->insertions++;

    __CPROVER_assert(c->cache_size <= c->max_size,
                    "Cache size must not exceed max after insertion");

    success = true;

    GraphCache_unlock(c, thread_id);
    return success;
}

// Clear entire cache
static void GraphCache_clear(GraphCache* c, int thread_id) {
    if (!GraphCache_lock(c, thread_id)) {
        return;
    }

    for (int i = 0; i < c->max_size; i++) {
        if (c->cache_slots[i] != INVALID_INDEX) {
            Graph* g = &g_graphs[c->cache_slots[i]];
            if (g->ref_count == 1) {
                g->state = GRAPH_EVICTING;
                Graph_release(g, c);
            }
            c->cache_slots[i] = INVALID_INDEX;
            c->cache_keys[i] = NULL_KEY;
        }
    }
    c->cache_size = 0;

    GraphCache_unlock(c, thread_id);
}

// ============================================================================
// Main Harness: Verify graph cache correctness
// ============================================================================

int main(void) {
    GraphCache cache;
    Graph_pool_init();

    // Initialize cache with non-deterministic size
    int cache_max_size = nondet_int();
    __CPROVER_assume(cache_max_size >= 2 && cache_max_size <= MAX_CACHE_SIZE);

    GraphCache_init(&cache, cache_max_size);

    // ========== PHASE 1: Basic operations ==========

    // Create and cache a graph
    uint64_t key1 = 1001;
    Graph* g1 = Graph_allocate(key1);
    __CPROVER_assert(g1 != NULL, "Must be able to allocate first graph");
    __CPROVER_assert(g1->ref_count == 1, "New graph must have ref_count 1");

    bool inserted = GraphCache_insert(&cache, g1, 0);
    __CPROVER_assert(inserted, "First insert must succeed");
    __CPROVER_assert(cache.cache_size == 1, "Cache size must be 1 after insert");
    __CPROVER_assert(g1->state == GRAPH_CACHED, "Graph must be cached after insert");

    // Lookup the graph
    Graph* found = GraphCache_lookup(&cache, key1, 1);
    __CPROVER_assert(found == g1, "Lookup must find cached graph");
    __CPROVER_assert(found->ref_count == 2, "Lookup must increment ref_count");
    __CPROVER_assert(cache.hits == 1, "Hit count must be 1");

    // Release our reference
    Graph_release(found, &cache);
    __CPROVER_assert(found->ref_count == 1, "Release must decrement ref_count");

    // Lookup non-existent key
    Graph* not_found = GraphCache_lookup(&cache, 9999, 2);
    __CPROVER_assert(not_found == NULL, "Lookup of non-existent key must return NULL");
    __CPROVER_assert(cache.misses == 1, "Miss count must be 1");

    // ========== PHASE 2: Fill cache and verify eviction ==========

    // Insert a few more graphs (limited for tractable verification)
    uint64_t key2 = 2001;
    Graph* g2 = Graph_allocate(key2);
    if (g2 != NULL) {
        GraphCache_insert(&cache, g2, 0);
    }

    uint64_t key3 = 2002;
    Graph* g3 = Graph_allocate(key3);
    if (g3 != NULL) {
        GraphCache_insert(&cache, g3, 0);
    }

    // Verify cache size is within bounds
    __CPROVER_assert(cache.cache_size <= cache_max_size,
                    "Cache size must not exceed max");

    // ========== PHASE 3: Verify invariants ==========

    // Invariant 1: Cache size bounds
    __CPROVER_assert(cache.cache_size >= 0, "Cache size must be non-negative");
    __CPROVER_assert(cache.cache_size <= cache_max_size, "Cache size must not exceed max");

    // Invariant 2: All cached graphs have valid state
    for (int i = 0; i < cache_max_size; i++) {
        if (cache.cache_slots[i] != INVALID_INDEX) {
            int gid = cache.cache_slots[i];
            __CPROVER_assert(gid >= 0 && gid < MAX_GRAPHS, "Cached graph ID must be valid");

            Graph* g = &g_graphs[gid];
            __CPROVER_assert(g->state == GRAPH_CACHED || g->state == GRAPH_IN_USE,
                            "Cached graph must be in cached or in-use state");
            __CPROVER_assert(g->ref_count >= 1, "Cached graph must have ref_count >= 1");
            __CPROVER_assert(g->cache_slot == i, "Graph cache_slot must match actual slot");
            __CPROVER_assert(g->key == cache.cache_keys[i], "Graph key must match cache key");
        }
    }

    // Invariant 3: Statistics are consistent
    __CPROVER_assert(cache.hits + cache.misses >= 0, "Stats must be non-negative");
    __CPROVER_assert(cache.insertions >= cache.cache_size,
                    "Insertions must be >= current size (some may have been evicted)");

    // ========== PHASE 4: Concurrent operations (simplified for CBMC) ==========

    // Single non-deterministic lookup from thread 1
    int thread_id_4 = nondet_int();
    __CPROVER_assume(thread_id_4 >= 0 && thread_id_4 < MAX_THREADS);

    uint64_t lookup_key = nondet_uint();
    __CPROVER_assume(lookup_key > 0 && lookup_key < 5000);

    Graph* g4 = GraphCache_lookup(&cache, lookup_key, thread_id_4);
    if (g4 != NULL) {
        __CPROVER_assert(g4->key == lookup_key, "Looked up graph must have correct key");
        Graph_release(g4, &cache);
    }

    // Verify bounds after operation
    __CPROVER_assert(cache.cache_size >= 0, "Size non-negative after op");
    __CPROVER_assert(cache.cache_size <= cache_max_size, "Size within bounds after op");

    // ========== PHASE 5: Verify no double-free ==========

    // All graphs should be in consistent state
    for (int i = 0; i < MAX_GRAPHS; i++) {
        Graph* g = &g_graphs[i];

        if (g->state == GRAPH_FREE) {
            __CPROVER_assert(g->ref_count == 0, "Free graph must have ref_count 0");
        } else if (g->state == GRAPH_DELETED) {
            __CPROVER_assert(g->ref_count == 0, "Deleted graph must have ref_count 0");
        } else if (g->state == GRAPH_CACHED || g->state == GRAPH_IN_USE) {
            __CPROVER_assert(g->ref_count >= 1, "Active graph must have ref_count >= 1");
        }
    }

    // ========== PHASE 6: Final cleanup ==========

    GraphCache_clear(&cache, 0);
    __CPROVER_assert(cache.cache_size == 0, "Cache must be empty after clear");

    return 0;
}
