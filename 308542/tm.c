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
#define READ_FLAG 0
#define WRITE_FLAG 1
#define REMOVED_FLAG 2
#define WRITE_REMOVE_FLAG 3
#define ADDED_FLAG 4
#define ADDED_REMOVED_FLAG 5

struct link {
    _Atomic(tx_t) status_owner;     // Identifier of the lock owner
    _Atomic(int) status;            // Whether this blocks need to be added or removed in case of rollback and commit

    struct link *prev;     // Previous link in the chain
    struct link *next;     // Next link in the chain

    size_t size;           // Size of the segment
};

/** Link reset.
 * @param link Link to reset
**/
static void link_reset(struct link *link, size_t size) {
    link->prev = link;
    link->next = link;
    link->size = size;
    link->status = DEFAULT_FLAG;
    link->status_owner = 0;
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

#define BATCHER_TRANSACTION_NUMBER 20
static const tx_t read_only_tx  = UINTPTR_MAX - 1;

struct batcher {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    uint32_t volatile counter;
    uint32_t volatile nb_entered;
    uint32_t volatile nb_write_tx;
};

struct region {
    void *start;        // Start of the shared memory region
    struct link allocs; // Allocated shared memory regions
    size_t size;        // Size of the shared memory region (in bytes)
    size_t align;       // Claimed alignment of the shared memory region (in bytes)
    size_t align_alloc; // Actual alignment of the memory allocations (in bytes)
    size_t delta_alloc; // Space to add at the beginning of the segment for the link chain (in bytes)

    struct batcher batcher;
};

// Return whether it's a success
bool init_batcher(struct batcher *batcher) {
    batcher->counter = BATCHER_TRANSACTION_NUMBER;
    batcher->nb_entered = 0;
    batcher->nb_write_tx = 0;
    if (pthread_cond_init(&(batcher->cond), NULL) != 0) {
        return false;
    }
    return pthread_mutex_init(&(batcher->mutex), NULL) == 0;
}

tx_t enter(struct batcher *batcher, bool is_ro) {
    tx_t tx = read_only_tx;
    // printf("Enter\n");
    pthread_mutex_lock(&(batcher->mutex));
    if (!is_ro) {
        while (batcher->counter == 0) {
            pthread_cond_wait(&(batcher->cond), &(batcher->mutex));
        }
        --batcher->counter;
        tx = ++batcher->nb_write_tx;
    }
    batcher->nb_entered++;
    // printf("Enter %d\n", batcher->nb_entered);
    pthread_mutex_unlock(&(batcher->mutex));
    return tx;
}

void batch_commit(struct region *region) {
    struct link *allocs = &(region->allocs);

    // Unlock each of our block
    struct link *link = &region->allocs;
    struct link *next_link = NULL;
    void *start = region->start;

    atomic_thread_fence(memory_order_acquire);
    while (true) {
        next_link = link->next;

        if (link->status_owner != 0 && (link->status == REMOVED_FLAG || link->status == ADDED_REMOVED_FLAG)) {
            // Free this block
            link_remove(link);
            free(link);
        } else {
            link->status = DEFAULT_FLAG;
            link->status_owner = 0;

            // Commit changes
            memcpy(start, ((char *) start) + link->size, link->size);

            // Reset locks
            memset(((char *) start) + 2 * link->size, 0, link->size / region->align * sizeof(tx_t));
        }

        if (next_link == allocs) {
            break;
        }

        link = next_link;
        start = (void *) (((char *) link) + region->delta_alloc);
    };

    atomic_thread_fence(memory_order_release);
}

void leave(struct batcher *batcher, struct region* region, tx_t tx) {
    pthread_mutex_lock(&(batcher->mutex));
    batcher->nb_entered--;
    // int d = batcher->nb_entered;
    if (batcher->nb_entered == 0) {
        if (batcher->nb_write_tx > 0) {
            batch_commit(region);
        } else {
            // printf("No commit needed\n");
        }
        batcher->nb_write_tx = 0;
        batcher->counter = BATCHER_TRANSACTION_NUMBER;
        pthread_cond_broadcast(&(batcher->cond));
        // printf("Batch commited\n");
    } else if (tx != read_only_tx) {
        pthread_cond_wait(&(batcher->cond), &(batcher->mutex));
    }
    pthread_mutex_unlock(&(batcher->mutex));
}

struct link *get_segment(const void *source, struct region *region, void **data_start) {
    struct link *allocs = &(region->allocs);

    *data_start = region->start;
    struct link *link = &region->allocs;

    while (true) {
        if (source >= *data_start && (char *) source < *(char **) data_start + link->size) {
            return link;
        }

        link = link->next;
        *data_start = (void *) ((char *) link + region->delta_alloc);

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

    size_t align_alloc = align < sizeof(void *) ? sizeof(void *) : align; // Also satisfy alignment requirement of 'struct link'
    size_t controle_size = size / align * sizeof(tx_t);

    if (unlikely(posix_memalign(&(region->start), align_alloc, 2 * size + controle_size) != 0)) {
        free(region);
        return invalid_shared;
    }

    if (unlikely(!init_batcher(&(region->batcher)))) {
        free(region);
        return invalid_shared;
    }

    // Do we store a pointer to the control location
    memset(region->start, 0, 2 * size + controle_size);
    link_reset(&(region->allocs), size);

    region->allocs.size = size;
    region->size = size;
    region->align = align;
    region->align_alloc = align_alloc;
    region->delta_alloc = (sizeof(struct link) + align_alloc - 1) / align_alloc * align_alloc;
    
    // printf("align %ld \n", align);
    // printf("align_alloc %ld \n", align_alloc);
    // printf("sizeof(struct link) %ld \n", sizeof(struct link));
    // printf("controle_size %ld \n", controle_size);
    // printf("delta_alloc %ld \n", region->delta_alloc);
    // printf("size %ld \n", size);
    // printf("getpagesize %d\n", getpagesize());
    // printf("sizeof char * %lu\n",  sizeof(char *));
    // printf("sizeof %lu\n",  sizeof(tx_t));
    // printf("sizeof atomic %lu\n",  sizeof(_Atomic(tx_t)));
    // printf("sizeof atomic %lu\n",  sizeof(_Atomic(uintptr_t)));
    // printf("sizeof atomic %lu\n",  sizeof(_Atomic(unsigned long long)));
    // return invalid_shared;

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
    tx_t tx = enter(&(((struct region *) shared)->batcher), is_ro);
    // if(tx == read_only_tx){
    //     printf("Begin read only\n");
    // } else {
    //     printf("Begin %lu\n", tx);
    // }
    return tx;
}

//TODO: Implement
void tm_rollback(shared_t shared, tx_t tx) {
    struct region *region = (struct region *) shared;
    struct link *allocs = &(region->allocs);

    // TODO: DEBUG line - print
    // __asm__ volatile ("int3;":::"memory");

    // Reverse each of our block
    struct link *link = &region->allocs;
    struct link *next_link = NULL;
    void *start = region->start;
    bool added = false;

    // Free allocated segments
    while (true) {
        next_link = link->next;
        added = false;

        tx_t owner = link->status_owner;
        if (owner == tx || owner == 0) {
            if (owner == tx && (link->status == ADDED_FLAG || link->status == ADDED_REMOVED_FLAG)) {
                link_remove(link);
                free(link);
                added = true;
            } else if (owner == tx) {
                link->status = DEFAULT_FLAG;
                link->status_owner = 0;
            }

            if (!added) {
                _Atomic(tx_t) volatile *controles = (_Atomic(tx_t) volatile *) ((char *) start + 2 * link->size);
                size_t align = region->align;
                size_t size = region->size;
                size_t nb = link->size / region->align;

                for (size_t i = 0; i < nb; ++i) {
                    if (controles[i] == tx) {
                        // printf("Unlock %ld\n", i);
                        memcpy(((char *) start) + i * align + size, ((char *) start) + i * align, align);
                        controles[i] = 0;
                    }
                }
                atomic_thread_fence(memory_order_release);
            }
        }

        if (next_link == allocs) {
            break;
        }

        link = next_link;
        start = (void *) (((char *) link) + region->delta_alloc);
    };
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

bool lock_words(struct region *region, tx_t tx, struct link *link, char* start, void *target, size_t size) {
    size_t index = ((char *) target - start) / region->align;
    size_t nb = size / region->align;
    // Not technicaly correct TODO: we should add some alignement between the data and the control structure
    _Atomic(tx_t) volatile *controls = (_Atomic(tx_t)volatile *) (start + link->size * 2);
    // volatile tx_t *controls = (volatile tx_t *) (start + link->size * 2);

    for (size_t i = index; i < index + nb; ++i) {
        tx_t previous = 0;
        bool res = atomic_compare_exchange_strong(controls + i, &previous, tx);
        if (!(res || previous == tx)) {
            if (i - index > 1) {
                // printf("Rollback i: %ld \t index: %ld\n", i-1, index);
                memset((void *)(controls + index), 0, (i - index - 1) * sizeof(tx_t));
                atomic_thread_fence(memory_order_release);
            }else {
                // printf("Unable to lock i: %ld \t index: %ld\n", i, index);
            }
            return false;
        } else {
            // printf("Lock %ld - %lu - %lu - %ld - %p\n", i, tx, size, nb, start);
        }
    }
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
    
    // VERSION 2
    if (tx == read_only_tx) {
        // Read the data
        memcpy(target, source, size);
    } else {
        void *data_start = NULL;
        struct link *link = get_segment(source, shared, &data_start);

        struct region *region = (struct region *) shared;
        size_t index = ((char *) source - (char*) data_start) / region->align;
        size_t nb = size / region->align;
        size_t align = region->align;

        // Not technical correct TODO: we should add some alignement between the data and the control structure
        _Atomic(tx_t) volatile *controls = (_Atomic(tx_t)volatile *) (data_start + link->size * 2);
        // volatile tx_t *controls = (volatile tx_t *) (data_start + link->size * 2);

        for (size_t i = index; i < index + nb; ++i) {
            if (controls[i] == tx) {
                memcpy(target, (char *) source + i * align + link->size, align);
            } else {
                memcpy(target, (char *) source + i * align, align);
            }
        }

        // if (!lock_words(shared, tx, link, data_start, source, size)) {
        //     tm_rollback(shared, tx);
        //     return false;
        // }

        // Read the data
        memcpy(target, (char *) source + link->size, size);
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
    void *data_start = NULL;
    struct link *link = get_segment(target, region, &data_start);

    if (!lock_words(region, tx, link, data_start, target, size)) {
        tm_rollback(shared, tx);
        return false;
    }

    memcpy((char *) target + link->size, source, size);
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

    size_t align_alloc = region->align_alloc;
    size_t delta_alloc = region->delta_alloc;
    size_t controle_size = size / region->align * sizeof(tx_t);
    void *segment;
    if (unlikely(posix_memalign(&segment, align_alloc,
                                delta_alloc + 2 * size + controle_size) !=
                 0)) // Allocation failed
        return nomem_alloc;

    // TODO: See with link_init() method
    struct link *link = segment;
    link->size = size;
    link->status_owner = tx;
    link->status = ADDED_FLAG;

    link_insert((struct link *) segment, &(((struct region *) shared)->allocs));
    segment = (void *) ((uintptr_t) segment + delta_alloc);
    memset(segment, 0, 2 * size + controle_size);
    *target = segment;

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
    size_t delta_alloc = ((struct region *) shared)->delta_alloc;
    struct link *link = (void *) ((uintptr_t) segment - delta_alloc);

    atomic_thread_fence(memory_order_acquire);
    tx_t owner = link->status_owner;
    if (unlikely(owner != 0 && owner != tx)) {
        // __asm__ volatile ("int3;":::"memory");
        // printf("%lu - owner %ld Unlikely lock free segment %p %d\n", tx, owner, segment, link->status);
        tm_rollback(shared, tx);
        return false;
    }

    link->status_owner = tx;

    if (link->status == ADDED_FLAG) {
        link->status = ADDED_REMOVED_FLAG;
    } else {
        link->status = REMOVED_FLAG;
    }

    return true;
}
