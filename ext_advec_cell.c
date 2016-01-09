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
 *  @brief C cell advection kernel.
 *  @author Wayne Gaudin
 *  @details Performs a second order advective remap using van-Leer limiting
 *  with directional splitting.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ext_chunk.h"
#include "ftocmacros.h"

void advec_cell_kernel_c_(int *xmin,int *xmax,int *ymin,int *ymax,
        int *dr,
        int *swp_nmbr,
        double *vertexdx,
        double *vertexdy,
        double *volume,
        double *density1,
        double *energy1,
        double *mass_flux_x,
        double *vol_flux_x,
        double *mass_flux_y,
        double *vol_flux_y,
        double *pre_vol,
        double *post_vol,
        double *pre_mass,
        double *post_mass,
        double *advec_vol,
        double *post_ener,
        double *ener_flux)
{
    int x_min=*xmin;
    int x_max=*xmax;
    int y_min=*ymin;
    int y_max=*ymax;
    int sweep_number=*swp_nmbr;
    int dir=*dr;
    int g_xdir=1,g_ydir=2;
    double one_by_six=1.0/6.0;
    int offload = _chunk.offload;

    START_PROFILING;

    if(dir == g_xdir)
    {
        if(sweep_number == 1)
        {
#pragma acc kernels loop independent collapse(2) if(offload) \
            present(pre_vol[:_chunk.bwid], post_vol[:_chunk.bwid], \
                    volume[:_chunk.wid], vol_flux_x[:_chunk.xwid], \
                    vol_flux_y[:_chunk.ywid])
//#pragma omp parallel for 
            for (int k = y_min-2; k <= y_max+2; k++)
            {
//#pragma ivdep
                for (int j = x_min-2; j <= x_max+2; j++)
                {
                    pre_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                        volume[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] +
                        (vol_flux_x[FTNREF2D(j+1,k  ,x_max+5,x_min-2,y_min-2)] -
                         vol_flux_x[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] +
                         vol_flux_y[FTNREF2D(j  ,k+1,x_max+4,x_min-2,y_min-2)] -
                         vol_flux_y[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)]);

                    post_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                        pre_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] -
                        (vol_flux_x[FTNREF2D(j+1,k  ,x_max+5,x_min-2,y_min-2)] -
                         vol_flux_x[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)]);
                }
            }
        }
        else 
        {
#pragma acc kernels loop independent collapse(2) if(offload) \
            present(pre_vol[:_chunk.bwid], post_vol[:_chunk.bwid], \
                    volume[:_chunk.wid], vol_flux_x[:_chunk.xwid], \
                    vol_flux_y[:_chunk.ywid])
//#pragma omp parallel for
            for (int k = y_min-2; k <= y_max+2; k++) 
            {
//#pragma ivdep
                for (int j = x_min-2; j <= x_max+2; j++) 
                {
                    pre_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                        volume[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] +
                        vol_flux_x[FTNREF2D(j+1,k  ,x_max+5,x_min-2,y_min-2)] -
                        vol_flux_x[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)]; 
                    post_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                        volume[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)];
                }
            }
        }
#pragma acc kernels loop independent collapse(2) if(offload) \
            present(pre_vol[:_chunk.bwid], post_vol[:_chunk.bwid], \
                    volume[:_chunk.wid], vol_flux_x[:_chunk.xwid], \
                    vol_flux_y[:_chunk.ywid], vertexdx[:_chunk.xmax+1], \
                    density1[:_chunk.wid], mass_flux_x[:_chunk.xwid], \
                    energy1[:_chunk.wid], ener_flux[:_chunk.bwid])
//#pragma omp parallel for 
        for (int k = y_min; k <= y_max; k++)
        {
            for (int j = x_min; j <= x_max+2; j++)
            {
                int pos = (vol_flux_x[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] > 0.0);
                int upwind = pos ? j-2 : MIN(j+1,x_max+2);
                int donor = pos ? j-1 : j;
                int downwind = pos ? j : j-1;
                int dif = pos ? donor : upwind;

                double sigmat = fabs(vol_flux_x[FTNREF2D(j,k,x_max+5,x_min-2,y_min-2)] /
                        pre_vol[FTNREF2D(donor,k  ,x_max+5,x_min-2,y_min-2)]);
                double sigma3 = (1.0+sigmat) * (vertexdx[FTNREF1D(j,x_min-2)] /
                        vertexdx[FTNREF1D(dif,x_min-2)]);
                double sigma4 = 2.0 - sigmat;

                double sigma = sigmat;
                double sigmav = sigmat;

                double diffuw = density1[FTNREF2D(donor,k  ,x_max+4,x_min-2,y_min-2)] -
                    density1[FTNREF2D(upwind,k  ,x_max+4,x_min-2,y_min-2)];
                double diffdw = density1[FTNREF2D(downwind,k  ,x_max+4,x_min-2,y_min-2)] -
                    density1[FTNREF2D(donor,k  ,x_max+4,x_min-2,y_min-2)];

                double limiter = (diffuw*diffdw > 0.0)*((1.0-sigmav)*SIGN(1.0,diffdw) *
                        MIN(fabs(diffuw), MIN(fabs(diffdw), 
                                one_by_six*(sigma3*fabs(diffuw)+sigma4*fabs(diffdw)))));

                mass_flux_x[FTNREF2D(j,k,x_max+5,x_min-2,y_min-2)] =
                    vol_flux_x[FTNREF2D(j,k,x_max+5,x_min-2,y_min-2)] *
                    (density1[FTNREF2D(donor,k  ,x_max+4,x_min-2,y_min-2)]+limiter);

                double sigmam = fabs(mass_flux_x[FTNREF2D(j,k,x_max+5,x_min-2,y_min-2)]) /
                    (density1[FTNREF2D(donor,k  ,x_max+4,x_min-2,y_min-2)] *
                     pre_vol[FTNREF2D(donor,k  ,x_max+5,x_min-2,y_min-2)]);
                diffuw = energy1[FTNREF2D(donor,k  ,x_max+4,x_min-2,y_min-2)] -
                    energy1[FTNREF2D(upwind,k  ,x_max+4,x_min-2,y_min-2)];
                diffdw = energy1[FTNREF2D(downwind,k  ,x_max+4,x_min-2,y_min-2)] -
                    energy1[FTNREF2D(donor,k  ,x_max+4,x_min-2,y_min-2)];

                limiter = (diffuw*diffdw>0.0)*((1.0-sigmam)*SIGN(1.0,diffdw) *
                        MIN(fabs(diffuw),MIN(fabs(diffdw),one_by_six *
                                (sigma3*fabs(diffuw)+sigma4*fabs(diffdw)))));

                ener_flux[FTNREF2D(j,k,x_max+5,x_min-2,y_min-2)] =
                    mass_flux_x[FTNREF2D(j,k,x_max+5,x_min-2,y_min-2)] *
                    (energy1[FTNREF2D(donor,k  ,x_max+4,x_min-2,y_min-2)]+limiter);
            }
        }

#pragma acc kernels loop independent collapse(2) if(offload) \
            present(pre_vol[:_chunk.bwid], vol_flux_x[:_chunk.xwid], \
                    density1[:_chunk.wid], mass_flux_x[:_chunk.xwid], \
                    energy1[:_chunk.wid], ener_flux[:_chunk.bwid])
//#pragma omp parallel for
        for (int k = y_min; k <= y_max; k++)
        {
//#pragma ivdep
            for (int j = x_min; j <= x_max; j++)
            {
                double pre_mass_s =
                    density1[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] *
                    pre_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)];
                double post_mass_s = pre_mass_s +
                    mass_flux_x[FTNREF2D(j  ,k,x_max+5,x_min-2,y_min-2)] -
                    mass_flux_x[FTNREF2D(j+1,k,x_max+5,x_min-2,y_min-2)];
                double post_ener_s =
                    (energy1[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] * pre_mass_s +
                     ener_flux[FTNREF2D(j  ,k,x_max+5,x_min-2,y_min-2)] -
                     ener_flux[FTNREF2D(j+1,k,x_max+5,x_min-2,y_min-2)]) / post_mass_s;
                double advec_vol_s =
                    pre_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] +
                    vol_flux_x[FTNREF2D(j  ,k,x_max+5,x_min-2,y_min-2)] -
                    vol_flux_x[FTNREF2D(j+1,k,x_max+5,x_min-2,y_min-2)];
                density1[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] = 
                    post_mass_s / advec_vol_s;
                energy1[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] = post_ener_s;
            }
        }
    }
    else if(dir == g_ydir)
    {
        if(sweep_number == 1)
        {
#pragma acc kernels loop independent collapse(2) if(offload) \
            present(pre_vol[:_chunk.bwid], post_vol[:_chunk.bwid], \
                    volume[:_chunk.wid], vol_flux_x[:_chunk.xwid], \
                    vol_flux_y[:_chunk.ywid])
//#pragma omp parallel for
            for (int k = y_min-2; k <= y_max+2; k++)
            {
//#pragma ivdep
                for (int j = x_min-2; j <= x_max+2; j++)
                {
                    pre_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                        volume[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] +
                        (vol_flux_y[FTNREF2D(j  ,k+1,x_max+4,x_min-2,y_min-2)] -
                         vol_flux_y[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] +
                         vol_flux_x[FTNREF2D(j+1,k  ,x_max+5,x_min-2,y_min-2)] -
                         vol_flux_x[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)]);
                    post_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                        pre_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] -
                        (vol_flux_y[FTNREF2D(j  ,k+1,x_max+4,x_min-2,y_min-2)] -
                         vol_flux_y[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)]);
                }
            }
        }
        else 
        {
#pragma acc kernels loop independent collapse(2) if(offload) \
            present(pre_vol[:_chunk.bwid], post_vol[:_chunk.bwid], \
                    volume[:_chunk.wid], vol_flux_x[:_chunk.xwid], \
                    vol_flux_y[:_chunk.ywid])
//#pragma omp parallel for 
            for (int k = y_min-2; k <= y_max+2; k++)
            {
//#pragma ivdep
                for (int j = x_min-2; j <= x_max+2; j++)
                {
                    pre_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                        volume[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] +
                        vol_flux_y[FTNREF2D(j  ,k+1,x_max+4,x_min-2,y_min-2)] -
                        vol_flux_y[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)];
                    post_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                        volume[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)];
                }
            }
        }

#pragma acc kernels loop independent collapse(2) if(offload) \
            present(pre_vol[:_chunk.bwid], post_vol[:_chunk.bwid], \
                    volume[:_chunk.wid], vol_flux_x[:_chunk.xwid], \
                    vol_flux_y[:_chunk.ywid], vertexdy[:_chunk.xmax+5], \
                    density1[:_chunk.wid], mass_flux_y[:_chunk.xwid], \
                    energy1[:_chunk.wid], ener_flux[:_chunk.bwid])
//#pragma omp parallel for
        for (int k = y_min; k <= y_max+2; k++)
        {
            for (int j = x_min; j <= x_max; j++)
            {
                int pos = (vol_flux_y[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] > 0.0);
                int upwind   = pos ? k-2 : MIN(k+1,y_max+2);
                int donor    = pos ? k-1 : k;
                int downwind = pos ? k : k-1;
                int dif      = pos ? donor : upwind;

                double sigmat = fabs(vol_flux_y[FTNREF2D(j,k,x_max+4,x_min-2,y_min-2)] /
                        pre_vol[FTNREF2D(j  ,donor,x_max+5,x_min-2,y_min-2)]);
                double sigma3 = (1.0+sigmat)*(vertexdy[FTNREF1D(k,y_min-2)] /
                        vertexdy[FTNREF1D(dif,y_min-2)]);
                double sigma4 = 2.0 - sigmat;

                double sigma=sigmat;
                double sigmav=sigmat;

                double diffuw = density1[FTNREF2D(j  ,donor,x_max+4,x_min-2,y_min-2)] - 
                    density1[FTNREF2D(j  ,upwind,x_max+4,x_min-2,y_min-2)];
                double diffdw = density1[FTNREF2D(j  ,downwind,x_max+4,x_min-2,y_min-2)] -
                    density1[FTNREF2D(j  ,donor,x_max+4,x_min-2,y_min-2)];

                double limiter = (diffuw*diffdw>0.0)*((1.0-sigmav)*SIGN(1.0,diffdw)*
                        MIN(fabs(diffuw),MIN(fabs(diffdw),
                                one_by_six*(sigma3*fabs(diffuw)+sigma4*fabs(diffdw)))));

                mass_flux_y[FTNREF2D(j,k,x_max+4,x_min-2,y_min-2)] =
                    vol_flux_y[FTNREF2D(j,k,x_max+4,x_min-2,y_min-2)] *
                    (density1[FTNREF2D(j  ,donor,x_max+4,x_min-2,y_min-2)]+limiter);

                double sigmam = fabs(mass_flux_y[FTNREF2D(j,k,x_max+4,x_min-2,y_min-2)]) /
                    (density1[FTNREF2D(j  ,donor,x_max+4,x_min-2,y_min-2)] *
                     pre_vol[FTNREF2D(j  ,donor,x_max+5,x_min-2,y_min-2)]);
                diffuw = energy1[FTNREF2D(j  ,donor,x_max+4,x_min-2,y_min-2)] -
                    energy1[FTNREF2D(j  ,upwind,x_max+4,x_min-2,y_min-2)];
                diffdw = energy1[FTNREF2D(j  ,downwind,x_max+4,x_min-2,y_min-2)] -
                    energy1[FTNREF2D(j  ,donor,x_max+4,x_min-2,y_min-2)];

                limiter = (diffuw*diffdw>0.0)*((1.0-sigmam)*SIGN(1.0,diffdw) *
                        MIN(fabs(diffuw),MIN(fabs(diffdw),one_by_six *
                                (sigma3*fabs(diffuw)+sigma4*fabs(diffdw)))));

                ener_flux[FTNREF2D(j,k,x_max+5,x_min-2,y_min-2)] =
                    mass_flux_y[FTNREF2D(j,k,x_max+4,x_min-2,y_min-2)] *
                    (energy1[FTNREF2D(j  ,donor,x_max+4,x_min-2,y_min-2)]+limiter);
            }
        }

#pragma acc kernels loop independent collapse(2) if(offload) \
            present(pre_vol[:_chunk.bwid], post_vol[:_chunk.bwid], \
                    volume[:_chunk.wid], vol_flux_x[:_chunk.xwid], \
                    vol_flux_y[:_chunk.ywid], vertexdx[:_chunk.xmax+5], \
                    density1[:_chunk.wid], mass_flux_y[:_chunk.xwid], \
                    energy1[:_chunk.wid], ener_flux[:_chunk.bwid])
//#pragma omp parallel for
        for (int k = y_min; k <= y_max; k++)
        {
//#pragma ivdep
            for (int j = x_min; j <= x_max; j++)
            {
                double pre_mass_s =
                    density1[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] *
                    pre_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)];
                double post_mass_s = pre_mass_s +
                    mass_flux_y[FTNREF2D(j,k  ,x_max+4,x_min-2,y_min-2)] -
                    mass_flux_y[FTNREF2D(j,k+1,x_max+4,x_min-2,y_min-2)];
                double post_ener_s =
                    (energy1[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] * pre_mass_s + 
                     ener_flux[FTNREF2D(j,k  ,x_max+5,x_min-2,y_min-2)] -
                     ener_flux[FTNREF2D(j,k+1,x_max+5,x_min-2,y_min-2)]) / post_mass_s;
                double advec_vol_s =
                    pre_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] +
                    vol_flux_y[FTNREF2D(j,k  ,x_max+4,x_min-2,y_min-2)] -
                    vol_flux_y[FTNREF2D(j,k+1,x_max+4,x_min-2,y_min-2)];
                density1[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] = 
                    post_mass_s / advec_vol_s;
                energy1[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] = post_ener_s;
            }
        }
    }

    STOP_PROFILING(__func__);
}

