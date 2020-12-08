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
#include <semaphore.h>

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

#if defined(USE_PTHREAD_LOCK)

struct lock_t {
    pthread_mutex_t mutex;
};

/** Initialize the given lock.
 * @param lock Lock to initialize
 * @return Whether the operation is a success
**/
static bool lock_init(struct lock_t* lock) {
    return pthread_mutex_init(&(lock->mutex), NULL) == 0;
}

/** Clean the given lock up.
 * @param lock Lock to clean up
**/
static void lock_cleanup(struct lock_t* lock) {
    pthread_mutex_destroy(&(lock->mutex));
}

/** Wait and acquire the given lock.
 * @param lock Lock to acquire
 * @return Whether the operation is a success
**/
static bool lock_acquire(struct lock_t* lock) {
    return pthread_mutex_lock(&(lock->mutex)) == 0;
}

/** Try to acquire the given lock.
 * @param lock Lock to acquire
 * @return Whether the operation is a success
**/
static bool lock_try_acquire(struct lock_t* lock) {
    return pthread_mutex_trylock(&(lock->mutex)) == 0;
}

/** Release the given lock.
 * @param lock Lock to release
**/
static void lock_release(struct lock_t* lock) {
    pthread_mutex_unlock(&(lock->mutex));
}

static bool lock_acquire_shared(struct lock_t* lock) {
    return lock_acquire(lock);
}

static bool lock_try_acquire_shared(struct lock_t* lock) {
    return lock_try_acquire(lock);
}

static void lock_release_shared(struct lock_t* lock) {
    lock_release(lock);
}

#elif defined(USE_TICKET_LOCK)

struct lock_t {
    atomic_ulong pass; // Ticket that acquires the lock
    atomic_ulong take; // Ticket the next thread takes
};

/** Initialize the given lock.
 * @param lock Lock to initialize
 * @return Whether the operation is a success
**/
static bool lock_init(struct lock_t* lock) {
    atomic_init(&(lock->pass), 0ul);
    atomic_init(&(lock->take), 0ul);
    return true;
}

/** Clean the given lock up.
 * @param lock Lock to clean up
**/
static void lock_cleanup(struct lock_t* lock as(unused)) {
    return;
}

/** Wait and acquire the given lock.
 * @param lock Lock to acquire
 * @return Whether the operation is a success
**/
static bool lock_acquire(struct lock_t* lock) {
    unsigned long ticket = atomic_fetch_add_explicit(&(lock->take), 1ul, memory_order_relaxed);
    while (atomic_load_explicit(&(lock->pass), memory_order_relaxed) != ticket)
        pause();
    atomic_thread_fence(memory_order_acquire);
    return true;
}

/** Release the given lock.
 * @param lock Lock to release
**/
static void lock_release(struct lock_t* lock) {
    atomic_fetch_add_explicit(&(lock->pass), 1, memory_order_release);
}

static bool lock_acquire_shared(struct lock_t* lock) {
    return lock_acquire(lock);
}

static void lock_release_shared(struct lock_t* lock) {
    lock_release(lock);
}

#elif defined(USE_RW_LOCK)

struct lock_t {
    pthread_rwlock_t rwlock;
};

/** Initialize the given lock.
 * @param lock Lock to initialize
 * @return Whether the operation is a success
**/
static bool lock_init(struct lock_t *lock) {
    return (0 == pthread_rwlock_init(&lock->rwlock, NULL));
}

/** Clean the given lock up.
 * @param lock Lock to clean up
**/
static void lock_cleanup(struct lock_t *lock as(unused)) {
    pthread_rwlock_destroy(&lock->rwlock);
}

/** Wait and acquire the given lock.
 * @param lock Lock to acquire
 * @return Whether the operation is a success
**/
static bool lock_acquire(struct lock_t *lock) {
    return (0 == pthread_rwlock_wrlock(&lock->rwlock));
}

/** Try to acquire the given lock.
 * @param lock Lock to acquire
 * @return Whether the operation is a success
**/
static bool lock_try_acquire(struct lock_t *lock) {
    return (0 == pthread_rwlock_trywrlock(&lock->rwlock));
}

/** Release the given lock.
 * @param lock Lock to release
**/
static void lock_release(struct lock_t *lock) {
    pthread_rwlock_unlock(&lock->rwlock);
}

/** Wait and acquire the given lock.
 * @param lock Lock to acquire
 * @return Whether the operation is a success
**/
static bool lock_acquire_shared(struct lock_t *lock) {
    return (0 == pthread_rwlock_rdlock(&lock->rwlock));
}

/** Wait and acquire the given lock.
 * @param lock Lock to acquire
 * @return Whether the operation is a success
**/
static bool lock_try_acquire_shared(struct lock_t *lock) {
    return (0 == pthread_rwlock_tryrdlock(&lock->rwlock));
}

/** Release the given lock.
 * @param lock Lock to release
**/
static void lock_release_shared(struct lock_t *lock) {
    pthread_rwlock_unlock(&lock->rwlock);
}

#else // Test-and-test-and-set

struct lock_t {
    atomic_bool locked; // Whether the lock is taken
};

/** Initialize the given lock.
 * @param lock Lock to initialize
 * @return Whether the operation is a success
**/
static bool lock_init(struct lock_t* lock) {
    atomic_init(&(lock->locked), false);
    return true;
}

/** Clean the given lock up.
 * @param lock Lock to clean up
**/
static void lock_cleanup(struct lock_t* lock as(unused)) {
    return;
}

/** Wait and acquire the given lock.
 * @param lock Lock to acquire
 * @return Whether the operation is a success
**/
static bool lock_acquire(struct lock_t* lock) {
    bool expected = false;
    while (unlikely(!atomic_compare_exchange_weak_explicit(&(lock->locked), &expected, true, memory_order_acquire, memory_order_relaxed))) {
        expected = false;
        while (unlikely(atomic_load_explicit(&(lock->locked), memory_order_relaxed)))
            pause();
    }
    return true;
}

/** Release the given lock.
 * @param lock Lock to release
**/
static void lock_release(struct lock_t* lock) {
    atomic_store_explicit(&(lock->locked), false, memory_order_release);
}

static bool lock_acquire_shared(struct lock_t* lock) {
    return lock_acquire(lock);
}

static void lock_release_shared(struct lock_t* lock) {
    lock_release(lock);
}

#endif

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
    struct link *prev;     // Previous link in the chain
    struct link *next;     // Next link in the chain
    struct lock_t lock;    // Lock
    size_t size;           // Size of the segment
    void *lock_owner;      // Identifier of the lock owner
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
    lock_init(&link->lock);
    link->lock_owner = NULL;
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

static const tx_t read_only_tx = UINTPTR_MAX - 10;
static const tx_t read_write_tx = UINTPTR_MAX - 11;

// -------------------------------------------------------------------------- //

#define BATCHER_TRANSACTION_NUMBER 100
struct batcher {
    // sem_t semaphore;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    uint32_t volatile counter;
    uint32_t volatile epoch;
    uint32_t volatile nbEntered;
};

struct region {
    // struct lock_t lock; // Global lock
    void *start;        // Start of the shared memory region
    struct link allocs; // Allocated shared memory regions
    size_t size;        // Size of the shared memory region (in bytes)
    size_t align;       // Claimed alignment of the shared memory region (in bytes)
    size_t align_alloc; // Actual alignment of the memory allocations (in bytes)
    size_t delta_alloc; // Space to add at the beginning of the segment for the link chain (in bytes)

    struct batcher batcher;
};

struct transaction {
    int id;
    bool is_ro;
    struct region *region;
    uint32_t epoch;
};

// Return wether it's a success
bool init_batcher(struct batcher *batcher) {
    batcher->epoch = 1;
    batcher->counter = BATCHER_TRANSACTION_NUMBER;
    batcher->nbEntered = 0;
    return pthread_mutex_init(&(batcher->mutex), NULL) == 0;
}

// Return the epoch
uint32_t enter(struct batcher *batcher) {
    // printf("Enter\n");
    pthread_mutex_lock(&(batcher->mutex));
    while (batcher->counter == 0) {
        pthread_cond_wait(&(batcher->cond), &(batcher->mutex));
    }
    batcher->counter--;
    batcher->nbEntered++;
    uint32_t epoch = batcher->epoch;
    pthread_mutex_unlock(&(batcher->mutex));
    return epoch;
}

void leave(struct batcher *batcher, struct transaction* tr) {
    pthread_mutex_lock(&(batcher->mutex));
    batcher->nbEntered--;
    if (batcher->nbEntered == 0){
        batch_commit(batcher->epoch, tr->region);
        batcher->epoch++;
        batcher->counter = BATCHER_TRANSACTION_NUMBER;
        pthread_cond_broadcast(&(batcher->cond));
    } else {
        pthread_cond_wait(&(batcher->cond), &(batcher->mutex));
    }
    pthread_mutex_unlock(&(batcher->mutex));
}

void batch_commit(uint32_t epoch, struct region *region) {
    struct link *allocs = &(region->allocs);

    // Unlock each of our block
    struct link *link = &region->allocs;
    struct link *next_link = NULL;
    void* start = region->start;
    // printf("--\n");
    while (true) {
        next_link = link->next;

        if (link->lock_owner != NULL) {
            // printf("Status %d\n", link->status);
            switch (link->status) {
                case WRITE_REMOVE_FLAG:
                case REMOVED_FLAG:
                    // Free this block
                    link_remove(link);
                    free(link);
                    break;
                case ADDED_FLAG:
                    // printf("Write %d\n", link->size);
                    // printf("Write link %x\n", link);
                    // printf("Write data %x\n", start);
                case WRITE_FLAG:
                    // Commit changes
                    memcpy(start, ((char*)start) + link->size, link->size);
                    // printf("after %x\n", start);
                    link->status = READ_FLAG;
                case READ_FLAG:
                    link->lock_owner = NULL;
                    lock_release(&link->lock);
                    break;
            }
            // printf("end %d\n", link->status);
        }
        if (next_link == allocs) {
            break;
        }

        link = next_link;
        start = (void*)(((uintptr_t)link) + region->delta_alloc);
    };
}

struct link *get_segment(void *source, struct transaction *tx, struct region *region, void **data_start) {
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

    if (unlikely(!init_batcher(&(region->batcher)))) {
        free(region);
        return invalid_shared;
    }
    memset(region->start, 0, size);
    link_reset(&(region->allocs));
    region->allocs.size = size;

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
    struct transaction *tr = malloc(sizeof(struct transaction));
    tr->is_ro = is_ro;
    tr->region = (struct region *)shared;
    tr->epoch = enter(&(tr->region->batcher));
    
    return (tx_t) tr;
    // if (is_ro) {
    //     if (unlikely(!lock_acquire_shared(&(((struct region*) shared)->lock))))
    //         return invalid_tx;
    //     return read_only_tx;
    // } else {
    //     if (unlikely(!lock_acquire(&(((struct region*) shared)->lock))))
    //         return invalid_tx;
    //     return read_write_tx;
    // }
}

//TODO: Implement
void tm_rollback(shared_t shared, tx_t tx) {
    // printf("Rollback start\n");
    struct region *region = (struct region *) shared;
    struct transaction *transaction = (struct transaction *) tx;
    struct link *allocs = &(region->allocs);

    // TODO: DEBUG line - print
    // __asm__ volatile ("int3;":::"memory");

    // Reverse each of our block
    struct link *link = &region->allocs;
    struct link *next_link = NULL;
    void *start = region->start;

    while (true) { // Free allocated segments
        next_link = link->next;

        if (link->lock_owner == transaction) {
            switch (link->status) {
                case WRITE_FLAG:
                case WRITE_REMOVE_FLAG:
                    // Restore previous data
                    memcpy(((char*)start) + link->size, start, link->size);
                case REMOVED_FLAG:
                    link->status = READ_FLAG;
                case READ_FLAG:
                    link->lock_owner = NULL;
                    lock_release(&link->lock);
                    break;
                case ADDED_FLAG:
                    link_remove(link);
                    free(link);
                    break;
            }
        }
        if (next_link == allocs) {
            // printf("Abort transaction\n");
            break;
        }

        link = next_link;
        start = (void*)(((uintptr_t)link) + region->delta_alloc);
    };
    leave(&(region->batcher), transaction);

    // printf("Rollback end\n");
}

/** [thread-safe] End the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to end
 * @return Whether the whole transaction committed
**/
bool tm_end(shared_t shared, tx_t tx) {
    leave(&((struct region*) shared)->batcher, tx);

    // printf("TM end tx %x\n", tx);
    // tm_commit(shared, tx);
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
    struct transaction *transaction = (struct transaction *) tx;
    void *data_start = NULL;
    struct link *link = get_segment(source, transaction, region, &data_start);

    // VERSION 2
    if (transaction->is_ro) {
        // Read the data
        memcpy(target, source, size);

    } else {
        // Lock acquire
        void *lockOwner = link->lock_owner;
        if (lockOwner == NULL) {
            if (!lock_try_acquire(&(link->lock))) {
                // printf("Read abort 1\n");
                tm_rollback(shared, tx);
                return false;
            }
            link->lock_owner = transaction;
        } else if (lockOwner != transaction) {
            // printf("Read abort 2\n");
            tm_rollback(shared, tx);
            return false;
        }
        
        // Read the data
        memcpy(target, source + link->size, size);
    }

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
bool tm_write(shared_t shared as(unused), tx_t tx as(unused), void const *source, size_t size, void *target) {
    struct region *region = (struct region *) shared;
    struct transaction *transaction = (struct transaction *) tx;
    void *data_start = NULL;
    struct link *link = get_segment(target, transaction, region, &data_start);

    // TODO: might be improved with compare&set in case of simple lock

    // VERSION 2
    // Lock acquire
    void *lockOwner = link->lock_owner;
    if (lockOwner == NULL) {
        if (!lock_try_acquire(&(link->lock))) {
            // printf("Write abort 1\n");
            tm_rollback(shared, tx);
            return false;
        }
        link->lock_owner = transaction;

    } else if (lockOwner != transaction) {
        // printf("Write miss-acquire link %x by tx %x\n", link, tx);
        // printf("Write abort 2\n");
        tm_rollback(shared, tx);
        return false;
    }

    link->status = WRITE_FLAG;

    // Write data
    // printf("Write acquire link %x by tx %x\n", link, tx);
    memcpy(target + link->size, source, size);
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
    struct transaction *transaction = (struct transaction *) tx;

    size_t align_alloc = ((struct region *) shared)->align_alloc;
    size_t delta_alloc = ((struct region *) shared)->delta_alloc;
    void *segment;
    if (unlikely(posix_memalign(&segment, align_alloc, delta_alloc + 2 * size) != 0)) // Allocation failed
        return nomem_alloc;

    // TODO: See with link_init() method
    struct link *link = segment;
    lock_init(&(link->lock));
    lock_acquire(&(link->lock));
    link->lock_owner = transaction;
    link->status = ADDED_FLAG;
    link->size = size;

    link_insert((struct link *) segment, &(((struct region *) shared)->allocs));
    segment = (void *) ((uintptr_t) segment + delta_alloc);
    memset(segment, 0, size);
    *target = segment;
    
    // printf("Link %x\n", link);
    // printf("Data %x\n", segment);
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
    struct transaction *transaction = (struct transaction *) tx;

    void *lockOwner = link->lock_owner;
    if (lockOwner == NULL) {
        if (!lock_try_acquire(&(link->lock))) {
            tm_rollback(shared, tx);
            return false;
        }
        link->lock_owner = transaction;

        // Save for eventual rollback
        link->status = REMOVED_FLAG;
    } else if (lockOwner != transaction) {
        tm_rollback(shared, tx);
        return false;
    } else if (link->status == WRITE_FLAG) {
        link->status = WRITE_REMOVE_FLAG;
    } else {
        link->status = REMOVED_FLAG;
    }

    return true;

    // Original code
//    size_t delta_alloc = ((struct region *) shared)->delta_alloc;
//    segment = (void *) ((uintptr_t) segment - delta_alloc);
//    link_remove((struct link *) segment);
//    free(segment);
//    return true;
}

