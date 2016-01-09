#include "ext_chunk.h"

CloverChunk _chunk;

void ext_init_(
        int* xmax,
        int* ymax,
        int* offload)
{
#ifdef OFFLOAD
    _chunk.offload = 1;
#else
    _chunk.offload = 0;
#endif
 
    _chunk.xmax = *xmax;
    _chunk.ymax = *ymax;
    _chunk.wid = (*xmax+4)*(*ymax+4);
    _chunk.bwid = (*xmax+5)*(*ymax+5);
    _chunk.xwid = (*xmax+5)*(*ymax+4);
    _chunk.ywid = (*xmax+4)*(*ymax+5);

    *offload = _chunk.offload;
}

void ext_finalise_()
{
    PRINT_PROFILING_RESULTS;
}
