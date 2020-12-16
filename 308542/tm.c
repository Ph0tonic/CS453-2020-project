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

// Requested features
#define _GNU_SOURCE
#define _POSIX_C_SOURCE   200809L
#ifdef __STDC_NO_ATOMICS__
#error Current C11 compiler does not support atomic operations
#endif

// External headers
// #include <unistd.h>
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

/** Compute a pointer to the parent structure.
 * @param ptr    Member pointer
 * @param type   Parent type
 * @param member Member name
 * @return Parent pointer
**/
#define objectof(ptr, type, member) \
    ((type*) ((uintptr_t) ptr - offsetof(type, member)))

#define DEFAULT_FLAG -1
#define REMOVED_FLAG 1
#define ADDED_FLAG 2
#define ADDED_REMOVED_FLAG 3

#define BATCHER_TRANSACTION_NUMBER 8
static const tx_t read_only_tx  = UINTPTR_MAX - 1;
static const tx_t destroy_tx  = UINTPTR_MAX - 2;

struct mapping_entry {
    void * ptr;
    _Atomic(tx_t) status_owner;     // Identifier of the lock owner
    _Atomic(int) status;            // Whether this blocks need to be added or removed in case of rollback and commit

    size_t size;           // Size of the segment
};

struct batcher {
    // pthread_mutex_t mutex;
    // pthread_cond_t cond;
    atomic_ulong counter;
    atomic_ulong nb_entered;
    atomic_ulong nb_write_tx;

    atomic_ulong pass; // Ticket that acquires the lock
    atomic_ulong take; // Ticket the next thread takes
    atomic_ulong epoch;
};

struct region {
    size_t size;        // Size of the shared memory region (in bytes)
    size_t align;       // Claimed alignment of the shared memory region (in bytes)
    size_t align_alloc; // Actual alignment of the memory allocations (in bytes)
    
    struct batcher batcher;

    struct mapping_entry *mapping;
    atomic_ulong index;
};

// Return whether it's a success
bool init_batcher(struct batcher *batcher) {
    batcher->counter = BATCHER_TRANSACTION_NUMBER;
    batcher->nb_entered = 0;
    batcher->nb_write_tx = 0;

    batcher->pass = 0;
    batcher->take = 0;
    batcher->epoch = 0;

    return true;
}

tx_t enter(struct batcher *batcher, bool is_ro) {
    if (is_ro) {
        // Acquire status lock
        unsigned long ticket = atomic_fetch_add_explicit(&(batcher->take), 1ul, memory_order_relaxed);
        while (atomic_load_explicit(&(batcher->pass), memory_order_relaxed) != ticket)
            pause();
        atomic_thread_fence(memory_order_acquire);

        atomic_fetch_add_explicit(&(batcher->nb_entered), 1ul, memory_order_relaxed);

        // Release status lock
        atomic_fetch_add_explicit(&(batcher->pass), 1ul, memory_order_release);
        // printf("enter readonly\n");
        return read_only_tx;
    } else {
        while(true) {
            unsigned long ticket = atomic_fetch_add_explicit(&(batcher->take), 1ul, memory_order_relaxed);
            while (atomic_load_explicit(&(batcher->pass), memory_order_relaxed) != ticket)
                pause();
            atomic_thread_fence(memory_order_acquire);

            // Acquire status lock
            if (batcher->counter == 0) {
                unsigned long int epoch = atomic_load_explicit(&(batcher->epoch), memory_order_relaxed);
                atomic_fetch_add_explicit(&(batcher->pass), 1ul, memory_order_release);

                while (atomic_load_explicit(&(batcher->epoch), memory_order_relaxed) == epoch)
                    pause();
                atomic_thread_fence(memory_order_acquire);
            } else {
                batcher->counter--;
                break;
            }
        }
        atomic_fetch_add_explicit(&(batcher->nb_entered), 1ul, memory_order_relaxed);
        atomic_fetch_add_explicit(&(batcher->pass), 1ul, memory_order_release);
        // printf("enter write\n");
        return atomic_fetch_add_explicit(&(batcher->nb_write_tx), 1ul, memory_order_relaxed) + 1ul;
    }
}

void batch_commit(struct region *region) {
    for (size_t i=0;i<region->index;++i) {
        struct mapping_entry *mapping = region->mapping + i;
        atomic_thread_fence(memory_order_acquire);
    
        if (mapping->status_owner != 0 && (mapping->status == REMOVED_FLAG || mapping->status == ADDED_REMOVED_FLAG)) {
            // Free this block
            unsigned long int previous = i+1;
            if (atomic_compare_exchange_strong(&(region->index), &previous, i)) {
                free(mapping->ptr);
            }
        } else if (mapping->status_owner != destroy_tx) {
            mapping->status = DEFAULT_FLAG;
            mapping->status_owner = 0;

            // Commit changes
            memcpy(mapping->ptr, ((char *) mapping->ptr) + mapping->size, mapping->size);

            // Reset locks
            memset(((char *) mapping->ptr) + 2 * mapping->size, 0, mapping->size / region->align * sizeof(tx_t));
        }
    };

    atomic_thread_fence(memory_order_release);
}

void leave(struct batcher *batcher, struct region* region, tx_t tx) {
    // Acquire status lock
    unsigned long ticket = atomic_fetch_add_explicit(&(batcher->take), 1ul, memory_order_relaxed);
    while (atomic_load_explicit(&(batcher->pass), memory_order_relaxed) != ticket)
        pause();
    atomic_thread_fence(memory_order_acquire);

    if (atomic_fetch_add_explicit(&batcher->nb_entered, -1ul, memory_order_relaxed) == 1ul) {
        if (atomic_load_explicit(&(batcher->nb_write_tx), memory_order_relaxed) > 0) {
            batch_commit(region);
            atomic_store_explicit(&(batcher->nb_write_tx), 0, memory_order_relaxed);
            atomic_store_explicit(&(batcher->counter), BATCHER_TRANSACTION_NUMBER, memory_order_relaxed);
            atomic_fetch_add_explicit(&(batcher->epoch), 1ul, memory_order_relaxed);
        }
        atomic_fetch_add_explicit(&(batcher->pass), 1ul, memory_order_release);
    } else if (tx != read_only_tx) {
        unsigned long int epoch = atomic_load_explicit(&(batcher->epoch), memory_order_relaxed);
        atomic_fetch_add_explicit(&(batcher->pass), 1ul, memory_order_release);

        while (atomic_load_explicit(&(batcher->epoch), memory_order_relaxed) == epoch)
            pause();
    } else {
        atomic_fetch_add_explicit(&(batcher->pass), 1ul, memory_order_release);
    }
}

struct mapping_entry *get_segment(const void *source, struct region *region, void **data_start) {
    for (size_t i=0;i<region->index;++i) {
        *data_start = region->mapping[i].ptr;
        if (source >= *data_start && (char *) source < *(char **) data_start + region->mapping[i].size) {
            return region->mapping + i;
        }
    }
    return NULL;
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

    size_t align_alloc = align < sizeof(void *) ? sizeof(void *) : align; // Also satisfy alignment requirement of 'struct link'
    size_t controle_size = size / align * sizeof(tx_t);

    region->index = 1;
    region->mapping = malloc(getpagesize());
    memset(region->mapping, 0, getpagesize());
    region->mapping->size = size;

    if (unlikely(posix_memalign(&(region->mapping->ptr), align_alloc, 2 * size + controle_size) != 0)) {
        free(region);
        return invalid_shared;
    }

    if (unlikely(!init_batcher(&(region->batcher)))) {
        free(region);
        return invalid_shared;
    }

    // Do we store a pointer to the control location
    memset(region->mapping->ptr, 0, 2 * size + controle_size);
    region->size = size;
    region->align = align;
    region->align_alloc = align_alloc;
    
    return region;
}

/** Destroy (i.e. clean-up + free) a given shared memory region.
 * @param shared Shared memory region to destroy, with no running transaction
**/
void tm_destroy(shared_t shared) {
    struct region *region = (struct region *) shared;
    
    for (size_t i=0;i<region->index;++i) {
        free(region->mapping[i].ptr);
    }
    free(region->mapping);
    free(region);
}

/** [thread-safe] Return the start address of the first allocated segment in the shared memory region.
 * @param shared Shared memory region to query
 * @return Start address of the first allocated segment
**/
void *tm_start(shared_t shared) {
    return ((struct region *) shared)->mapping->ptr;
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
    return enter(&(((struct region *) shared)->batcher), is_ro);
}

void tm_rollback(shared_t shared, tx_t tx) {
    struct region *region = (struct region *)shared;
    unsigned long int index = region->index;
    for (size_t i=0;i<index;++i) {
        bool added = false;
        struct mapping_entry *mapping = region->mapping + i;
        added = false;

        tx_t owner = mapping->status_owner;
        if (owner == tx || owner == 0) {
            if (owner == tx && (mapping->status == ADDED_FLAG || mapping->status == ADDED_REMOVED_FLAG)) {
                mapping->status_owner = destroy_tx;
                added = true;
            } else if (owner == tx) {
                mapping->status_owner = 0;
                mapping->status = DEFAULT_FLAG;
            }

            if (!added) {
                _Atomic(tx_t) volatile *controles = (_Atomic(tx_t) volatile *) ((char *)mapping->ptr + 2 * mapping->size);
                size_t align = region->align;
                size_t size = region->size;
                size_t nb = mapping->size / region->align;

                for (size_t i = 0; i < nb; ++i) {
                    if (controles[i] == tx) {
                        // printf("Unlock %ld\n", i);
                        memcpy(((char *) mapping->ptr) + i * align + size, ((char *) mapping->ptr) + i * align, align);
                        controles[i] = 0;
                    }
                }
                atomic_thread_fence(memory_order_release);
            }
        }
    };
    // printf("Rollback\n");
    leave(&(region->batcher), region, tx);
    // printf("Rollback end\n");
}

/** [thread-safe] End the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to end
 * @return Whether the whole transaction committed
**/
bool tm_end(shared_t shared, tx_t tx) {
    leave(&((struct region *) shared)->batcher, (struct region *)shared, tx);
    return true;
}

bool lock_words(struct region *region, tx_t tx, struct mapping_entry *mapping, char* start, void *target, size_t size) {
    size_t index = ((char *) target - start) / region->align;
    size_t nb = size / region->align;

    _Atomic(tx_t) volatile *controls = (_Atomic(tx_t)volatile *) (start + mapping->size * 2);
    // volatile tx_t *controls = (volatile tx_t *) (start + link->size * 2);

    for (size_t i = index; i < index + nb; ++i) {
        tx_t previous = 0;
        bool res = atomic_compare_exchange_strong(controls + i, &previous, tx);
        if (!(res || previous == tx)) {
            if (i - index > 1) {
                // printf("Rollback i: %ld \t index: %ld\n", i-1, index);
                memset((void *)(controls + index), 0, (i - index - 1) * sizeof(tx_t));
                atomic_thread_fence(memory_order_release);
            }
            return false;
        }
    }
    return true;
}

void tm_read_write(shared_t shared, tx_t tx, void const *source, size_t size, void *target) {
    void *data_start = NULL;
    struct mapping_entry *mapping = get_segment(source, shared, &data_start);

    struct region *region = (struct region *) shared;
    size_t index = ((char *) source - (char*) data_start) / region->align;
    size_t nb = size / region->align;
    size_t align = region->align;

    _Atomic(tx_t) volatile *controls = (_Atomic(tx_t)volatile *) (data_start + mapping->size * 2);
    // volatile tx_t *controls = (volatile tx_t *) (data_start + link->size * 2);

    for (size_t i = index; i < index + nb; ++i) {
        if (controls[i] == tx) {
            memcpy(target, (char *) source + i * align + mapping->size, align);
        } else {
            memcpy(target, (char *) source + i * align, align);
        }
    }

    // Read the data
    memcpy(target, (char *) source + mapping->size, size);
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
    if (likely(tx == read_only_tx)) {
        // Read the data
        memcpy(target, source, size);
    } else {
        tm_read_write(shared, tx, source, size, target);
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
bool tm_write(shared_t shared, tx_t tx, void const *source, size_t size, void *target) {
    struct region *region = (struct region *) shared;
    void *data_start = NULL;
    struct mapping_entry *mapping = get_segment(target, region, &data_start);
    
    if (mapping == NULL || !lock_words(region, tx, mapping, data_start, target, size)) {
        tm_rollback(shared, tx);
        return false;
    }

    memcpy((char *) target + mapping->size, source, size);
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
    struct region *region = (struct region *) shared;

    unsigned long int index = atomic_fetch_add_explicit(&(region->index), 1ul, memory_order_relaxed);

    struct mapping_entry *mapping = region->mapping + index;
    mapping->size = size;
    mapping->status_owner = tx;
    mapping->status = ADDED_FLAG;

    size_t align_alloc = region->align_alloc;
    size_t controle_size = size / region->align * sizeof(tx_t);
    
    if (unlikely(posix_memalign(&mapping->ptr, align_alloc, 2 * size + controle_size) != 0)) // Allocation failed
        return nomem_alloc;

    memset(mapping->ptr, 0, 2 * size + controle_size);
    *target = mapping->ptr;

    // printf("Link %x\n", link);
    // printf("%lu Alloc %p\n", tx, segment);
    return success_alloc;
}

/** [thread-safe] Memory freeing in the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param target Address of the first byte of the previously allocated segment to deallocate
 * @return Whether the whole transaction can continue
**/
bool tm_free(shared_t shared, tx_t tx, void *segment) {
    void *data_start = NULL;
    
    struct mapping_entry *mapping = get_segment(segment, shared, &data_start);
    tx_t previous = 0;

    if (mapping == NULL || !(atomic_compare_exchange_strong(&mapping->status_owner, &previous, tx) || previous == tx)) {
        // __asm__ volatile ("int3;":::"memory");
        // printf("%lu - owner %ld Unlikely lock free segment %p %d\n", tx, owner, segment, link->status);
        tm_rollback(shared, tx);
        return false;
    }

    if (mapping->status == ADDED_FLAG) {
        mapping->status = ADDED_REMOVED_FLAG;
    } else {
        mapping->status = REMOVED_FLAG;
    }

    return true;
}
