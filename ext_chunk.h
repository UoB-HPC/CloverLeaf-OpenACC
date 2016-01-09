#ifndef __CHUNK
#define __CHUNK

// Global chunk class
#include "ext_profiler.h"

typedef struct
{
    int offload;
    int wid;
    int bwid;
    int xwid;
    int ywid;
    int xmax;
    int ymax;
} CloverChunk;

extern CloverChunk _chunk;

#endif
