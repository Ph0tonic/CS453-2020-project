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
#include <unistd.h>
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
// static inline void pause() {
// #if (defined(__i386__) || defined(__x86_64__)) && defined(USE_MM_PAUSE)
//     _mm_pause();
// #else
//     sched_yield();
// #endif
// }

// -------------------------------------------------------------------------- //

/** Compute a pointer to the parent structure.
 * @param ptr    Member pointer
 * @param type   Parent type
 * @param member Member name
 * @return Parent pointer
**/
#define objectof(ptr, type, member) \
    ((type*) ((uintptr_t) ptr - offsetof(type, member)))

// -------------------------------------------------------------------------- //

// TODO: see with https://stackoverflow.com/questions/31402979/how-to-get-number-of-cores-on-linux-using-c-programming/31405118
#define BATCHER_TRANSACTION_NUMBER 8

static const tx_t read_only_tx  = UINTPTR_MAX - 1;
static const tx_t destroy_tx  = UINTPTR_MAX - 2;

//TODO: Remove useless flags
#define DEFAULT_FLAG 0ul
#define REMOVED_FLAG 1ul
#define ADDED_REMOVED_FLAG 2ul
#define ADDED_FLAG 3ul

static inline bool is_destroyed(tx_t status) {
    return status == destroy_tx;
}

static inline bool is_unchanged(tx_t status) {
    return status == DEFAULT_FLAG;
}

static inline bool is_added(tx_t status) {
    return (status - 1) % 3 >= ADDED_REMOVED_FLAG - 1;
}

static inline bool is_removed(tx_t status) {
    return (status - 1) % 3 <= ADDED_REMOVED_FLAG - 1;
}

static inline bool is_owner(tx_t status, tx_t tx) {
    return !is_unchanged(status) && !is_destroyed(status) && (status - 1) / 3 == tx;
}

static inline tx_t status_for_tx(tx_t tx, unsigned long int status) {
    return (tx - 1) * 3 + status;
}

struct mapping_entry {
    void* ptr;
    tx_t status_owner;     // Identifier of the lock owner
    size_t size;
};

struct batcher {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    uint32_t volatile counter;
    uint32_t volatile nb_entered;
    uint32_t volatile nb_write_tx;
};

struct region {
    size_t size;        // Size of the shared memory region (in bytes)
    size_t align;       // Claimed alignment of the shared memory region (in bytes)
    size_t align_alloc; // Actual alignment of the memory allocations (in bytes)

    struct batcher batcher;

    _Atomic(uint64_t) index;
    struct mapping_entry *data_mapping;
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
    for (size_t i=0;i<region->index;++i) {
        struct mapping_entry *mapping = region->data_mapping + i;
        tx_t owner = mapping->status_owner;
        if (!is_unchanged(owner) && is_removed(owner)) {
            // Free this block
            mapping->status_owner = destroy_tx;
            if (region->index == i+1) {
                free(mapping->ptr);
                --region->index;
            }

            //TODO: we might be able to improve efficiency in case of latest release segment.
        } else if (!is_destroyed(owner)) {
            mapping->status_owner = DEFAULT_FLAG;

            // Commit changes
            memcpy(mapping->ptr, ((char *) mapping->ptr) + mapping->size, mapping->size);

            // Reset locks
            memset(((char *) mapping->ptr) + 2 * mapping->size, 0, mapping->size / region->align_alloc * sizeof(tx_t));
        }
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

struct mapping_entry *get_segment(const void *source, struct region *region, void **data_start) {
    //TODO: adapt if 32 bits
    uint64_t index = ((uint64_t)source >> 32) - 1;
    uint64_t offset = (uint64_t)source & 0x00000000FFFFFFFF;
    // printf("Index: %lu \tOffset: %p\t source %p\n",index, offset, source);

    //TODO: Check me
    struct mapping_entry *mapping = region->data_mapping + index;
    *data_start = (void *)((char *)mapping->ptr + offset);

    // if ((uint64_t)(*data_start) % region->align != 0) {
    //     __asm__ volatile ("int3;":::"memory");
    // }

    return mapping;
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

    // TODO: Alloc data_mapping
    region->index = 1;
    region->data_mapping = malloc(2*getpagesize());
    region->data_mapping[0].size = size;
    memset(region->data_mapping, 0, sizeof(getpagesize()));

    if (unlikely(posix_memalign(&(region->data_mapping[0].ptr), align_alloc, 2 * size + controle_size) != 0)) {
        free(region);
        return invalid_shared;
    }

    // Init first segment
    memset(region->data_mapping[0].ptr, 0, 2 * size + controle_size);

    if (unlikely(!init_batcher(&(region->batcher)))) {
        free(region);
        return invalid_shared;
    }

    // region->allocs.size = size;
    region->size = size;
    region->align = align;
    region->align_alloc = align_alloc;

    // printf("align %ld \n", align);
    // printf("align_alloc %ld \n", align_alloc);
    // printf("controle_size %ld \n", controle_size);
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
    // printf("TM_DESTROY !!!");
    for (size_t i=0;i<region->index;++i) {
        free(region->data_mapping[i].ptr);
    }
    free(region->data_mapping);
    free(region);
}

/** [thread-safe] Return the start address of the first allocated segment in the shared memory region.
 * @param shared Shared memory region to query
 * @return Start address of the first allocated segment
**/
void *tm_start(shared_t shared as(unused)) {
    // printf("Start get %p\n", (void*)(1ul << 32));
    return (void*)(1ul << 32);
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
    return ((struct region *) shared)->align_alloc;
}

/** [thread-safe] Begin a new transaction on the given shared memory region.
 * @param shared Shared memory region to start a transaction on
 * @param is_ro  Whether the transaction is read-only
 * @return Opaque transaction ID, 'invalid_tx' on failure
**/
tx_t tm_begin(shared_t shared, bool is_ro) {
    return enter(&(((struct region *) shared)->batcher), is_ro);
}

//TODO: Implement
void tm_rollback(shared_t shared, tx_t tx) {
    struct region *region = (struct region *) shared;

    // TODO: DEBUG line - print
    // __asm__ volatile ("int3;":::"memory");

    // Reverse each of our block
    size_t nb = region->index;
    for(size_t i=0;i<nb;++i) {
        struct mapping_entry *mapping = region->data_mapping + i;

        tx_t owner = mapping->status_owner;

        if (is_owner(owner, tx) || is_unchanged(owner)) {
            bool added = is_owner(owner, tx) && is_added(owner);
            if (added) {
                mapping->status_owner = destroy_tx;
            } else if (is_owner(owner, tx)) {
                mapping->status_owner = 0;
            }

            if (!added) {
                size_t size = mapping->size;
                size_t align = region->align_alloc;
                size_t nb = mapping->size / align;
                _Atomic(tx_t) volatile *controles = (_Atomic(tx_t) volatile *) ((char *) mapping->ptr + 2 * size);

                for (size_t j = 0; j < nb; ++j) {
                    if (controles[j] == tx) {
                        // printf("Unlock %ld\n", i);
                        memcpy(((char *) mapping->ptr) + j * align + size, ((char *) mapping->ptr) + j * align, align);
                        controles[j] = 0;
                    }
                }
                atomic_thread_fence(memory_order_release);
            }
        }
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

bool lock_words(struct region *region, tx_t tx, struct mapping_entry *mapping, void *target, size_t size) {
    size_t index = ((char *)target - (char *)mapping->ptr) / region->align_alloc;
    size_t nb = size / region->align_alloc;

    _Atomic(tx_t) volatile *controls = (_Atomic(tx_t)volatile *) (mapping->ptr + mapping->size * 2);
    
    for (size_t i = index; i < index + nb; ++i) {
        tx_t previous = 0;
        if (!(atomic_compare_exchange_strong(controls + i, &previous, tx) || previous == tx)) {
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
    
    
    if (tx == read_only_tx) {
        // Read the data
        memcpy(target, (char*)region->data_mapping[((uint64_t)source >> 32) - 1].ptr + ((size_t)source & 0x00000000FFFFFFFF), size);
    } else {
        void *data_start = NULL;
        struct mapping_entry *mapping = get_segment(source, region, &data_start);
        
        // printf("###%p %p, %p, %p\n",source, mapping, mapping->ptr, data_start);
        size_t align = region->align_alloc;
        size_t index = ((char *)data_start - (char*)mapping->ptr) / align;
        size_t nb = size / align;

        // Not technical correct TODO: we should add some alignement between the data and the control structure
        _Atomic(tx_t) volatile *controls = (_Atomic(tx_t)volatile *) (mapping->ptr + mapping->size * 2);
        
        for (size_t i = index; i < index + nb; ++i) {
            if (controls[i] == tx) {
                memcpy(target, (char *) data_start + i * align + mapping->size, align);
            } else {
                memcpy(target, (char *) data_start + i * align, align);
            }
        }

        // if (!lock_words(shared, tx, link, data_start, source, size)) {
        //     tm_rollback(shared, tx);
        //     return false;
        // }

        // Read the data
        memcpy(target, (char *) data_start + mapping->size, size);
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

    if (!lock_words(region, tx, mapping, data_start, size)) {
        tm_rollback(shared, tx);
        return false;
    }

    memcpy((char *)data_start + mapping->size, source, size);
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

    // TODO: get new index
    uint64_t id = atomic_fetch_add(&region->index, 1);
    struct mapping_entry *mapping = region->data_mapping + id;
    mapping->size = size;

    size_t align_alloc = region->align_alloc;
    size_t controle_size = size / region->align * sizeof(tx_t);
    // printf("Alloc index %lu: %p\n", id);
    if (unlikely(posix_memalign(&mapping->ptr, align_alloc, 2 * size + controle_size) != 0)) // Allocation failed
        return nomem_alloc;

    memset(mapping->ptr, 0, 2 * size + controle_size);
    mapping->status_owner = status_for_tx(tx, ADDED_FLAG);

    //TODO: adapt if 32 bits
    *target = (void*)((id + 1) << 32);

    return success_alloc;
}

/** [thread-safe] Memory freeing in the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param target Address of the first byte of the previously allocated segment to deallocate
 * @return Whether the whole transaction can continue
**/
bool tm_free(shared_t shared, tx_t tx, void *segment) {
    struct region *region = (struct region *) shared;
    //TODO: Adapt for 32 bits
    uint64_t index = ((uint64_t)segment >> 32) - 1;
    
    // printf("Free index %lu\n", index);

    tx_t previous = 0;
    if (!(atomic_compare_exchange_strong(&region->data_mapping[index].status_owner, &previous, status_for_tx(tx, REMOVED_FLAG)) || is_owner(previous, tx))) {
        // printf("%lu - owner %ld Unlikely lock free segment %p %d\n", tx, owner, segment, link->status);
        tm_rollback(shared, tx);
        return false;
    }

    if (is_added(region->data_mapping[index].status_owner)) {
        region->data_mapping[index].status_owner = status_for_tx(tx, ADDED_REMOVED_FLAG);
    }

    return true;
}
