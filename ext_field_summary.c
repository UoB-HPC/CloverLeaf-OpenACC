/*Crown Copyright 2012 AWE.
 *
 * This file is part of CloverLeaf.
 *
 * CloverLeaf is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * CloverLeaf is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/**
 *  @brief C field summary kernel
 *  @author Wayne Gaudin
 *  @details The total mass, internal energy, kinetic energy and volume weighted
 *  pressure for the chunk is calculated.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ftocmacros.h"
#include "ext_chunk.h"

void field_summary_kernel_c_(int *xmin,
        int *xmax,
        int *ymin,
        int *ymax,
        double *volume,
        double *density0,
        double *energy0,
        double *pressure,
        double *xvel0,
        double *yvel0,
        double *vl,
        double *mss,
        double *ien,
        double *ken,
        double *prss,
        int* offload)
{
    int x_min=*xmin;
    int x_max=*xmax;
    int y_min=*ymin;
    int y_max=*ymax;

    double vol = 0.0;
    double mass = 0.0;
    double ie = 0.0;
    double ke = 0.0;
    double press = 0.0;

    int field_offload = _chunk.offload && *offload;

    START_PROFILING;

#pragma acc kernels if(field_offload) \
    present(xvel0[:_chunk.bwid], yvel0[:_chunk.bwid], volume[:_chunk.wid],\
            density0[:_chunk.wid], pressure[:_chunk.wid], energy0[:_chunk.wid]) 
#pragma acc loop independent \
    reduction(+ : vol, mass, press, ie, ke)
    for (int k = y_min; k <= y_max; k++) 
    {
#pragma acc loop independent
        for (int j = x_min;j <= x_max; j++) 
        {
            double vsqrd=0.0;
            for (int kv = k; kv <= k+1; kv++) 
            {
                for (int jv = j; jv <= j+1; jv++) 
                {
                    vsqrd = vsqrd + 0.25 * (
                            xvel0[FTNREF2D(jv ,kv ,x_max+5,x_min-2,y_min-2)] *
                            xvel0[FTNREF2D(jv ,kv ,x_max+5,x_min-2,y_min-2)] +
                            yvel0[FTNREF2D(jv ,kv ,x_max+5,x_min-2,y_min-2)] *
                            yvel0[FTNREF2D(jv ,kv ,x_max+5,x_min-2,y_min-2)]);
                }
            }

            double cell_vol = volume[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)];
            double cell_mass = cell_vol*density0[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)];
            vol = vol+cell_vol;
            mass = mass+cell_mass;
            ie = ie+cell_mass*energy0[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)];
            ke = ke+cell_mass*0.5*vsqrd;
            press = press+cell_vol*pressure[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)];
        }
    }

    *vl=vol;
    *mss=mass;
    *ien=ie;
    *ken=ke;
    *prss=press;

    STOP_PROFILING(__func__);
}
