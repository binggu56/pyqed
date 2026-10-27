#include <malloc/malloc.h>
#include <mach/mach.h>
#include <stddef.h>
struct usage { size_t bytes, blocks; };
static kern_return_t read_self(task_t task, vm_address_t address, vm_size_t size, void **memory) {
    (void)task; (void)size; *memory=(void*)address; return KERN_SUCCESS;
}
static void count_ranges(task_t task, void *context, unsigned type, vm_range_t *ranges, unsigned count) {
    (void)task;
    struct usage *usage=context;
    if (!(type & MALLOC_PTR_IN_USE_RANGE_TYPE)) return;
    for (unsigned i=0;i<count;++i) { usage->bytes+=ranges[i].size; ++usage->blocks; }
}
int allocated_ranges(size_t *bytes, size_t *blocks) {
    vm_address_t *zones=NULL; unsigned count=0;
    struct usage usage={0,0};
    kern_return_t status=malloc_get_all_zones(mach_task_self(),read_self,&zones,&count);
    if (status!=KERN_SUCCESS) return status;
    for (unsigned i=0;i<count;++i) {
        malloc_zone_t *zone=(malloc_zone_t*)zones[i];
        if (!zone->introspect || !zone->introspect->enumerator) return -1;
        status=zone->introspect->enumerator(mach_task_self(),&usage,MALLOC_PTR_IN_USE_RANGE_TYPE,zones[i],read_self,count_ranges);
        if (status!=KERN_SUCCESS) return status;
    }
    *bytes=usage.bytes; *blocks=usage.blocks; return 0;
}
