/**
 * @file   tm.c
 * @author Wermeille Bastien <bastien.wermeille@epfl.ch>
 * @version 1.0
 *
 * @section LICENSE
 *
 * [...]
 *
 * @section DESCRIPTION
 *
 * Implementation of my wonderful transaction manager.
**/

// Compile-time configuration
// #define USE_MM_PAUSE
#define USE_PTHREAD_LOCK
// #define USE_TICKET_LOCK
// #define USE_RW_LOCK

// Requested features
#define _GNU_SOURCE
#define _POSIX_C_SOURCE   200809L
#ifdef __STDC_NO_ATOMICS__
#error Current C11 compiler does not support atomic operations
#endif

// External headers
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h> //TODO: REMOVE FOR DEBUG PURPOSES


#if (defined(__i386__) || defined(__x86_64__)) && defined(USE_MM_PAUSE)
#include <xmmintrin.h>
#else

#include <sched.h>

#endif

// Internal headers
#include <tm.h>

// -------------------------------------------------------------------------- //

/** Define a proposition as likely true.
 * @param prop Proposition
**/
#undef likely
#ifdef __GNUC__
#define likely(prop) \
        __builtin_expect((prop) ? 1 : 0, 1)
#else
#define likely(prop) \
        (prop)
#endif

/** Define a proposition as likely false.
 * @param prop Proposition
**/
#undef unlikely
#ifdef __GNUC__
#define unlikely(prop) \
        __builtin_expect((prop) ? 1 : 0, 0)
#else
#define unlikely(prop) \
        (prop)
#endif

/** Define one or several attributes.
 * @param type... Attribute names
**/
#undef as
#ifdef __GNUC__
#define as(type...) \
        __attribute__((type))
#else
#define as(type...)
#warning This compiler has no support for GCC attributes
#endif

// -------------------------------------------------------------------------- //

/** Pause for a very short amount of time.
**/
static inline void pause() {
#if (defined(__i386__) || defined(__x86_64__)) && defined(USE_MM_PAUSE)
    _mm_pause();
#else
    sched_yield();
#endif
}

// -------------------------------------------------------------------------- //

/** Compute a pointer to the parent structure.
 * @param ptr    Member pointer
 * @param type   Parent type
 * @param member Member name
 * @return Parent pointer
**/
#define objectof(ptr, type, member) \
    ((type*) ((uintptr_t) ptr - offsetof(type, member)))

#define READ_FLAG 0
#define WRITE_FLAG 1
#define REMOVED_FLAG 2
#define WRITE_REMOVE_FLAG 3
#define ADDED_FLAG 4

struct link {
    volatile tx_t owner;           // Identifier of the lock owner
    struct link *prev;     // Previous link in the chain
    struct link *next;     // Next link in the chain
    size_t size;           // Size of the segment
    uint8_t status;        // Whether this blocks need to be added or removed in case of rollback and commit
//    uint32_t ts;           // Timestamp
//    uint32_t old_ts;       // Timestamp save

    // TODO: add additional write info data
};

/** Link reset.
 * @param link Link to reset
**/
static void link_reset(struct link *link) {
    link->prev = link;
    link->next = link;
    link->owner = 0;
    link->status = READ_FLAG;
}

/** Link insertion before a "base" link.
 * @param link Link to insert
 * @param base Base link relative to which 'link' will be inserted
**/
static void link_insert(struct link *link, struct link *base) {
    struct link *prev = base->prev;
    link->prev = prev;
    link->next = base;
    base->prev = link;
    prev->next = link;
}

/** Link removal.
 * @param link Link to remove
**/
static void link_remove(struct link *link) {
    struct link *prev = link->prev;
    struct link *next = link->next;
    prev->next = next;
    next->prev = prev;
}

// -------------------------------------------------------------------------- //

static const tx_t read_only_tx  = UINTPTR_MAX - 1;

struct region {
    // struct lock_t lock; // Global lock
    void *start;        // Start of the shared memory region
    struct link allocs; // Allocated shared memory regions
    size_t size;        // Size of the shared memory region (in bytes)
    size_t align;       // Claimed alignment of the shared memory region (in bytes)
    size_t align_alloc; // Actual alignment of the memory allocations (in bytes)
    size_t delta_alloc; // Space to add at the beginning of the segment for the link chain (in bytes)

    atomic_int counter; // TX identifier counter
};

// struct transaction {
//     int id;
//     bool is_ro;
//     struct region *region;
// };

struct link *get_segment(const void *source, struct region *region, void **data_start) {
    struct link *allocs = &(region->allocs);

    *data_start = region->start;
    struct link *link = &region->allocs;

    while (true) {
        if (source >= *data_start && source < *data_start + link->size) {
            return link;
        }

        link = link->next;
        *data_start = (void *)((uintptr_t)link + region->delta_alloc);

        if (link == allocs) {
            return NULL; // Not found
        }
    };
}

/** Create (i.e. allocate + init) a new shared memory region, with one first non-free-able allocated segment of the requested size and alignment.
 * @param size  Size of the first shared segment of memory to allocate (in bytes), must be a positive multiple of the alignment
 * @param align Alignment (in bytes, must be a power of 2) that the shared memory region must support
 * @return Opaque shared memory region handle, 'invalid_shared' on failure
**/
shared_t tm_create(size_t size, size_t align) {
    struct region *region = (struct region *) malloc(sizeof(struct region));
    if (unlikely(!region)) {
        return invalid_shared;
    }
    size_t align_alloc =
            align < sizeof(void *) ? sizeof(void *) : align; // Also satisfy alignment requirement of 'struct link'
    if (unlikely(posix_memalign(&(region->start), align_alloc, 2 * size) != 0)) {
        free(region);
        return invalid_shared;
    }
    // if (unlikely(!lock_init(&(region->lock)))) {
    //     free(region->start);
    //     free(region);
    //     return invalid_shared;
    // }
    memset(region->start, 0, size);
    link_reset(&(region->allocs));
    region->allocs.size = size;
    region->counter = 1;

    region->size = size;
    region->align = align;
    region->align_alloc = align_alloc;
    region->delta_alloc = (sizeof(struct link) + align_alloc - 1) / align_alloc * align_alloc;
    return region;
}

/** Destroy (i.e. clean-up + free) a given shared memory region.
 * @param shared Shared memory region to destroy, with no running transaction
**/
void tm_destroy(shared_t shared) {
    struct region *region = (struct region *) shared;
    struct link *allocs = &(region->allocs);
    while (true) { // Free allocated segments
        struct link *alloc = allocs->next;
        if (alloc == allocs)
            break;
        link_remove(alloc);
        free(alloc);
    }
    free(region->start);
    free(region);
    // lock_cleanup(&(region->lock));
}

/** [thread-safe] Return the start address of the first allocated segment in the shared memory region.
 * @param shared Shared memory region to query
 * @return Start address of the first allocated segment
**/
void *tm_start(shared_t shared) {
    return ((struct region *) shared)->start;
}

/** [thread-safe] Return the size (in bytes) of the first allocated segment of the shared memory region.
 * @param shared Shared memory region to query
 * @return First allocated segment size
**/
size_t tm_size(shared_t shared) {
    return ((struct region *) shared)->size;
}

/** [thread-safe] Return the alignment (in bytes) of the memory accesses on the given shared memory region.
 * @param shared Shared memory region to query
 * @return Alignment used globally
**/
size_t tm_align(shared_t shared) {
    return ((struct region *) shared)->align;
}

/** [thread-safe] Begin a new transaction on the given shared memory region.
 * @param shared Shared memory region to start a transaction on
 * @param is_ro  Whether the transaction is read-only
 * @return Opaque transaction ID, 'invalid_tx' on failure
**/
tx_t tm_begin(shared_t shared, bool is_ro) {
    struct region *region = (struct region *) shared;

    int id = atomic_fetch_add_explicit(&(region->counter), 1, memory_order_relaxed);
    
    return (tx_t) id;
}

//TODO: Implement
void tm_rollback(struct region *region, tx_t transaction) {
    // printf("Rollback start\n");
    struct link *allocs = &(region->allocs);

    // TODO: DEBUG line - print
    // __asm__ volatile ("int3;":::"memory");

    // Reverse each of our block
    void *start = region->start;
    struct link *link = &region->allocs;

    struct link *next_link = NULL;
    while (true) { // Free allocated segments
        next_link = link->next;

        if (atomic_load(&link->owner) == transaction) {
            switch (link->status) {
                case WRITE_FLAG:
                    __attribute__ ((fallthrough));
                case WRITE_REMOVE_FLAG:
                    // Restore previous data
                    memcpy(start, ((char*)start) + link->size, link->size);
                    __attribute__ ((fallthrough));
                case REMOVED_FLAG:
                    link->status = READ_FLAG;
                    __attribute__ ((fallthrough));
                case READ_FLAG:
                    link->owner = 0;
                    break;
                case ADDED_FLAG:
                    link_remove(link);
                    free(link);
                    break;
            }
        }
        if (next_link == allocs) {
            break;
        }

        link = next_link;
        start = link + region->delta_alloc;
    };
    // printf("Rollback end\n");
}

void tm_commit(shared_t shared, tx_t tx) {
    // printf("Commit start\n");
    struct region *region = (struct region *) shared;
    struct link *allocs = &(region->allocs);

    // Unlock each of our block
    struct link *link = &region->allocs;
    struct link *next_link = NULL;

    while (true) {
        next_link = link->next;

        if (atomic_load(&link->owner) == tx) {
            switch (link->status) {
                case WRITE_REMOVE_FLAG:
                    __attribute__ ((fallthrough));
                case REMOVED_FLAG:
                    // Free this block
                    link_remove(link);
                    free(link);
                    break;
                case ADDED_FLAG:
                    __attribute__ ((fallthrough));
                case WRITE_FLAG:
                    link->status = READ_FLAG;
                    __attribute__ ((fallthrough));
                case READ_FLAG:
                    link->owner = 0;
                    break;
            }
        }
        if (next_link == allocs) {
            break;
        }

        link = next_link;
    };
    // printf("Commit end\n");
}

/** [thread-safe] End the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to end
 * @return Whether the whole transaction committed
**/
bool tm_end(shared_t shared, tx_t tx) {
    // printf("TM end tx %x\n", tx);
    tm_commit(shared, tx);
    return true;
}

/** [thread-safe] Read operation in the given transaction, source in the shared region and target in a private region.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param source Source start address (in the shared region)
 * @param size   Length to copy (in bytes), must be a positive multiple of the alignment
 * @param target Target start address (in a private region)
 * @return Whether the whole transaction can continue
**/
bool tm_read(shared_t shared, tx_t tx, void const *source, size_t size, void *target) {
    struct region *region = (struct region *) shared;
    void *data_start = NULL;
    struct link *link = get_segment(source, region, &data_start);

    // Lock acquire
    tx_t previous = 0;
    if (atomic_load(&link->owner) != tx && !atomic_compare_exchange_strong(&link->owner, &previous, tx)) {
        tm_rollback(region, tx);
        return false;
    }

    // Read the data
    memcpy(target, source, size);

    // __asm__ volatile ("int3;":::"memory");
    return true;
}

/** [thread-safe] Write operation in the given transaction, source in a private region and target in the shared region.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param source Source start address (in a private region)
 * @param size   Length to copy (in bytes), must be a positive multiple of the alignment
 * @param target Target start address (in the shared region)
 * @return Whether the whole transaction can continue
**/
bool tm_write(shared_t shared, tx_t tx, void const *source, size_t size, void *target) {
    struct region *region = (struct region *) shared;
    void *data_start = NULL;
    struct link *link = get_segment(target, region, &data_start);

    // Lock acquire    
    tx_t previous = 0;
    if (atomic_load(&link->owner) != tx && !atomic_compare_exchange_strong(&link->owner, &previous, tx)) {
        tm_rollback(region, tx);
        return false;
    }

    if (link->status == READ_FLAG) {
        // Update status
        link->status = WRITE_FLAG;

        // Save segment before write
        memcpy(((char*)data_start) + link->size, data_start, link->size);
    }

    // Write data
    memcpy(target, source, size);
    return true;
}

/** [thread-safe] Memory allocation in the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param size   Allocation requested size (in bytes), must be a positive multiple of the alignment
 * @param target Pointer in private memory receiving the address of the first byte of the newly allocated, aligned segment
 * @return Whether the whole transaction can continue (success/nomem), or not (abort_alloc)
**/
alloc_t tm_alloc(shared_t shared, tx_t tx, size_t size, void **target) {
    size_t align_alloc = ((struct region *) shared)->align_alloc;
    size_t delta_alloc = ((struct region *) shared)->delta_alloc;
    void *segment;
    if (unlikely(posix_memalign(&segment, align_alloc, delta_alloc + 2 * size) != 0)) // Allocation failed
        return nomem_alloc;

    // TODO: See with link_init() method
    struct link *link = segment;
    link->owner = tx;
    link->status = ADDED_FLAG;
    link->size = size;

    link_insert((struct link *) segment, &(((struct region *) shared)->allocs));
    segment = (void *) ((uintptr_t) segment + delta_alloc);
    memset(segment, 0, size);
    *target = segment;
    return success_alloc;
}

/** [thread-safe] Memory freeing in the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param target Address of the first byte of the previously allocated segment to deallocate
 * @return Whether the whole transaction can continue
**/
bool tm_free(shared_t shared, tx_t tx, void *segment) {
    size_t delta_alloc = ((struct region *) shared)->delta_alloc;
    struct link *link = (void *) ((uintptr_t) segment - delta_alloc);
    
    tx_t previous = 0;
    if (atomic_load(&link->owner) != tx && !atomic_compare_exchange_strong(&link->owner, &previous, tx)) {
        tm_rollback((struct region *)shared, tx);
        return false;
    }
    
    if (link->status == WRITE_FLAG) {
        link->status = WRITE_REMOVE_FLAG;
    } else {
        link->status = REMOVED_FLAG;
    }

    return true;
}
