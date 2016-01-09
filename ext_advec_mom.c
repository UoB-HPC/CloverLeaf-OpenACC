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
 *  @brief C momentum advection kernel
 *  @author Wayne Gaudin
 *  @details Performs a second order advective remap on the vertex momentum
 *  using van-Leer limiting and directional splitting.
 *  Note that although pre_vol is only set and not used in the update, please
 *  leave it in the method.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ftocmacros.h"
#include "ext_chunk.h"

void advec_mom_kernel_c_(int *xmin,int *xmax,int *ymin,int *ymax,
        double* restrict vel1,
        double* restrict mass_flux_x,
        double* restrict vol_flux_x,
        double* restrict mass_flux_y,
        double* restrict vol_flux_y,
        double* restrict volume,
        double* restrict density1,
        double* restrict node_flux,
        double* restrict node_mass_post,
        double* restrict node_mass_pre,
        double* restrict mom_flux,
        double* restrict pre_vol,
        double* restrict post_vol,
        double* restrict celldx,
        double* restrict celldy,
        int *whch_vl,
        int *swp_nmbr,
        int *drctn)

{
    int x_min=*xmin;
    int x_max=*xmax;
    int y_min=*ymin;
    int y_max=*ymax;
    int which_vel=*whch_vl;
    int sweep_number=*swp_nmbr;
    int direction=*drctn;
    int mom_sweep;
    int upwind,donor,downwind,dif;
    double sigma,wind,width;
    double vdiffuw,vdiffdw,auw,adw,limiter;
    int offload = _chunk.offload;

    double advec_vel_s;

    mom_sweep=direction+2*(sweep_number-1);

    START_PROFILING;

    if(mom_sweep==1)
    {
#pragma acc parallel if(offload) \
            present(pre_vol[:_chunk.bwid], post_vol[:_chunk.bwid], \
                    volume[:_chunk.wid], vol_flux_x[:_chunk.xwid], \
                    vol_flux_y[:_chunk.ywid])
#pragma acc loop independent collapse(2)
        for (int k = y_min-2; k <= y_max+2; k++)
        {
            for (int j = x_min-2; j <= x_max+2; j++)
            {
                post_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                    volume[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] +
                    vol_flux_y[FTNREF2D(j  ,k+1,x_max+4,x_min-2,y_min-2)] -
                    vol_flux_y[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)];
                pre_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                    post_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] +
                    vol_flux_x[FTNREF2D(j+1,k  ,x_max+5,x_min-2,y_min-2)] -
                    vol_flux_x[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)];
            }
        }
    } 
    else if(mom_sweep==2)
    {
#pragma acc parallel if(offload) \
            present(pre_vol[:_chunk.bwid], post_vol[:_chunk.bwid], \
                    volume[:_chunk.wid], vol_flux_x[:_chunk.xwid], \
                    vol_flux_y[:_chunk.ywid])
#pragma acc loop independent collapse(2) 
//#pragma omp parallel for
        for (int k = y_min-2; k <= y_max+2; k++)
        {
//#pragma ivdep
            for (int j = x_min-2; j <= x_max+2; j++)
            {
                post_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                    volume[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] +
                    vol_flux_x[FTNREF2D(j+1,k  ,x_max+5,x_min-2,y_min-2)] -
                    vol_flux_x[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)];
                pre_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                    post_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] +
                    vol_flux_y[FTNREF2D(j  ,k+1,x_max+4,x_min-2,y_min-2)] -
                    vol_flux_y[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)];
            }
        }
    } 
    else if(mom_sweep==3)
    {
#pragma acc parallel if(offload) \
            present(pre_vol[:_chunk.bwid], post_vol[:_chunk.bwid], \
                    volume[:_chunk.wid], vol_flux_x[:_chunk.xwid], \
                    vol_flux_y[:_chunk.ywid])
#pragma acc loop independent collapse(2) 
//#pragma omp parallel for 
        for (int k = y_min-2; k <= y_max+2; k++)
        {
//#pragma ivdep
            for (int j = x_min-2; j <= x_max+2; j++)
            {
                post_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                    volume[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)];
                pre_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                    post_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] +
                    vol_flux_y[FTNREF2D(j  ,k+1,x_max+4,x_min-2,y_min-2)] -
                    vol_flux_y[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)];
            }
        }
    } 
    else if(mom_sweep==4)
    {
#pragma acc parallel if(offload) \
            present(pre_vol[:_chunk.bwid], post_vol[:_chunk.bwid], \
                    volume[:_chunk.wid], vol_flux_x[:_chunk.xwid], \
                    vol_flux_y[:_chunk.ywid])
#pragma acc loop independent collapse(2) 
//#pragma omp parallel for 
        for (int k = y_min-2; k <= y_max+2; k++)
        {
//#pragma ivdep
            for (int j = x_min-2; j <= x_max+2; j++)
            {
                post_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                    volume[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)]; 
                pre_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                    post_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] + 
                    vol_flux_x[FTNREF2D(j+1,k  ,x_max+5,x_min-2,y_min-2)] -
                    vol_flux_x[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)];
            }
        }
    }

    if(direction == 1) 
    {
        if(which_vel == 1)
        {
#pragma acc parallel if(offload) \
            present(node_flux[:_chunk.bwid], mass_flux_x[:_chunk.xwid])
#pragma acc loop independent collapse(2) 
            for (int k = y_min; k <= y_max+1; k++) 
            {
                for (int j = x_min-2; j <= x_max+2; j++) 
                {
                    node_flux[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)]=0.25
                        *(mass_flux_x[FTNREF2D(j  ,k-1,x_max+5,x_min-2,y_min-2)]
                                +mass_flux_x[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)]
                                +mass_flux_x[FTNREF2D(j+1,k-1,x_max+5,x_min-2,y_min-2)]
                                +mass_flux_x[FTNREF2D(j+1,k  ,x_max+5,x_min-2,y_min-2)]);
                }
            }
#pragma acc parallel if(offload) \
            present(node_mass_post[:_chunk.bwid], post_vol[:_chunk.bwid],\
                    density1[:_chunk.wid], node_flux[:_chunk.bwid], \
                    node_mass_pre[:_chunk.bwid])
#pragma acc loop independent collapse(2) 
//#pragma omp parallel for 
            for (int k=y_min;k<=y_max+1;k++) 
            {
//#pragma ivdep
                for (int j=x_min-1;j<=x_max+2;j++) 
                {
                    node_mass_post[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)]=0.25
                        *(density1[FTNREF2D(j  ,k-1,x_max+4,x_min-2,y_min-2)]
                                *post_vol[FTNREF2D(j  ,k-1,x_max+5,x_min-2,y_min-2)]
                                +density1[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)]
                                *post_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)]
                                +density1[FTNREF2D(j-1,k-1,x_max+4,x_min-2,y_min-2)]
                                *post_vol[FTNREF2D(j-1,k-1,x_max+5,x_min-2,y_min-2)]
                                +density1[FTNREF2D(j-1,k  ,x_max+4,x_min-2,y_min-2)]
                                *post_vol[FTNREF2D(j-1,k  ,x_max+5,x_min-2,y_min-2)]);

                    node_mass_pre[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                        node_mass_post[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] -
                        node_flux[FTNREF2D(j-1,k  ,x_max+5,x_min-2,y_min-2)] +
                        node_flux[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)];
                }
            }
        }

#pragma acc parallel if(offload) \
            present(celldx[:_chunk.xmax], post_vol[:_chunk.bwid], \
                    vel1[:_chunk.bwid], mom_flux[:_chunk.bwid], \
                    node_mass_pre[:_chunk.bwid], node_flux[:_chunk.bwid])
#pragma acc loop independent collapse(2) 
//#pragma omp parallel for 
        for (int k = y_min; k <= y_max+1; k++)
        {
            for (int j = x_min-1; j <= x_max+1; j++)
            {
                int pos = (node_flux[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] < 0.0);
                int upwind = pos ? j+2 : j-1;
                int donor = pos ? j+1 : j;
                int downwind = pos ? j : j+1;
                int dif = pos ? donor : upwind;

                double sigma = fabs(node_flux[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)]) /
                    (node_mass_pre[FTNREF2D(donor,k  ,x_max+5,x_min-2,y_min-2)]);
                double width = celldx[FTNREF1D(j,x_min-2)];
                double vdiffuw = vel1[FTNREF2D(donor,k  ,x_max+5,x_min-2,y_min-2)] -
                    vel1[FTNREF2D(upwind,k  ,x_max+5,x_min-2,y_min-2)];
                double vdiffdw = vel1[FTNREF2D(downwind,k  ,x_max+5,x_min-2,y_min-2)] -
                    vel1[FTNREF2D(donor,k  ,x_max+5,x_min-2,y_min-2)];

                double limiter=0.0;
                if(vdiffuw*vdiffdw > 0.0)
                {
                    double auw = fabs(vdiffuw);
                    double adw = fabs(vdiffdw);
                    double wind = (vdiffdw <= 0.0) ? -1.0 : 1.0;
                    limiter = wind * 
                        MIN(width*((2.0-sigma)*adw/width+(1.0+sigma)*auw 
                                    / celldx[FTNREF1D(dif,x_min-2)])/6.0, MIN(auw,adw));
                }

                double advec_vel_s = (1.0-sigma)*limiter + 
                    vel1[FTNREF2D(donor,k  ,x_max+5,x_min-2,y_min-2)];
                mom_flux[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] = advec_vel_s *
                    node_flux[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)];
            }
        }
#pragma acc parallel if(offload) \
            present(celldx[:_chunk.xmax], post_vol[:_chunk.bwid], \
                    vel1[:_chunk.bwid], mom_flux[:_chunk.bwid], \
                    node_mass_post[:_chunk.bwid], node_mass_pre[:_chunk.bwid])
#pragma acc loop independent collapse(2) 
//#pragma omp parallel for 
        for (int k=y_min;k<=y_max+1;k++) 
        {
//#pragma ivdep
            for (int j=x_min;j<=x_max+1;j++) 
            {
                vel1[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                    (vel1[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] *
                     node_mass_pre[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] +
                     mom_flux[FTNREF2D(j-1,k  ,x_max+5,x_min-2,y_min-2)] -
                     mom_flux[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)]) /
                    node_mass_post[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)];
            }
        }
    }
    else if(direction==2)
    {
        if(which_vel == 1)
        {
#pragma acc parallel if(offload) \
            present(mass_flux_y[:_chunk.ywid], node_flux[:_chunk.bwid])
#pragma acc loop independent collapse(2) 
//#pragma omp parallel for 
            for (int k=y_min-2;k<=y_max+2;k++) 
            {
//#pragma ivdep
                for (int j=x_min;j<=x_max+1;j++) 
                {
                    node_flux[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] = 0.25 *
                        (mass_flux_y[FTNREF2D(j-1,k  ,x_max+4,x_min-2,y_min-2)] +
                         mass_flux_y[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)] +
                         mass_flux_y[FTNREF2D(j-1,k+1,x_max+4,x_min-2,y_min-2)] +
                         mass_flux_y[FTNREF2D(j  ,k+1,x_max+4,x_min-2,y_min-2)]);
                }
            }
#pragma acc parallel if(offload) \
            present(post_vol[:_chunk.bwid], density1[:_chunk.wid], \
                    node_flux[:_chunk.bwid], node_mass_pre[:_chunk.bwid], \
                    node_mass_post[:_chunk.bwid])
#pragma acc loop independent collapse(2) 
//#pragma omp parallel for 
            for (int k=y_min-1;k<=y_max+2;k++) {
//#pragma ivdep
                for (int j=x_min;j<=x_max+1;j++) {
                    node_mass_post[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)]=0.25
                        *(density1[FTNREF2D(j  ,k-1,x_max+4,x_min-2,y_min-2)]
                                *post_vol[FTNREF2D(j  ,k-1,x_max+5,x_min-2,y_min-2)]
                                +density1[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)]
                                *post_vol[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)]
                                +density1[FTNREF2D(j-1,k-1,x_max+4,x_min-2,y_min-2)]
                                *post_vol[FTNREF2D(j-1,k-1,x_max+5,x_min-2,y_min-2)]
                                +density1[FTNREF2D(j-1,k  ,x_max+4,x_min-2,y_min-2)]
                                *post_vol[FTNREF2D(j-1,k  ,x_max+5,x_min-2,y_min-2)]);

                    node_mass_pre[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                        node_mass_post[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] -
                        node_flux[FTNREF2D(j  ,k-1,x_max+5,x_min-2,y_min-2)] +
                        node_flux[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)];
                }
            }
        }

#pragma acc parallel if(offload) \
            present(node_flux[:_chunk.bwid], post_vol[:_chunk.bwid], \
                    vel1[:_chunk.bwid], mom_flux[:_chunk.bwid], \
                    node_mass_pre[:_chunk.bwid], celldy[:_chunk.ymax])
#pragma acc loop independent collapse(2) 
//#pragma omp parallel for
        for (int k=y_min-1;k<=y_max+1;k++) 
        {
            for (int j=x_min;j<=x_max+1;j++) 
            {
                int pos = (node_flux[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] < 0.0);
                int upwind   = pos ? k+2 : k-1;
                int donor    = pos ? k+1 : k;
                int downwind = pos ? k : k+1;
                int dif      = pos ? donor : upwind;

                double sigma = fabs(node_flux[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)]) /
                    (node_mass_pre[FTNREF2D(j  ,donor,x_max+5,x_min-2,y_min-2)]);
                double width = celldy[FTNREF1D(k,y_min-2)];
                double vdiffuw = vel1[FTNREF2D(j  ,donor,x_max+5,x_min-2,y_min-2)] -
                    vel1[FTNREF2D(j  ,upwind,x_max+5,x_min-2,y_min-2)];
                double vdiffdw = vel1[FTNREF2D(j  ,downwind ,x_max+5,x_min-2,y_min-2)] -
                    vel1[FTNREF2D(j  ,donor,x_max+5,x_min-2,y_min-2)];

                double limiter=0.0;

                if(vdiffuw*vdiffdw>0.0)
                {
                    double auw = fabs(vdiffuw);
                    double adw = fabs(vdiffdw);
                    double wind = 1.0;
                    if(vdiffdw <= 0.0) 
                    {
                        wind = -1.0;
                    }

                    limiter = wind*MIN(width*((2.0-sigma)*adw/width+(1.0+sigma)
                                *auw/celldy[FTNREF1D(dif,y_min-2)])/6.0,MIN(auw,adw));
                }

                double advec_vel_s = (1.0-sigma)*limiter + 
                    vel1[FTNREF2D(j  ,donor,x_max+5,x_min-2,y_min-2)];
                mom_flux[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] = advec_vel_s*
                    node_flux[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)];
            }
        }

#pragma acc parallel if(offload) \
        present(vel1[:_chunk.bwid], mom_flux[:_chunk.bwid], \
                node_mass_post[:_chunk.bwid], node_mass_pre[:_chunk.bwid])
#pragma acc loop independent collapse(2) 
        for (int k=y_min;k<=y_max+1;k++) 
        {
            for (int j=x_min;j<=x_max+1;j++) 
            {
                vel1[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] =
                    (vel1[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] *
                     node_mass_pre[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)] +
                     mom_flux[FTNREF2D(j  ,k-1,x_max+5,x_min-2,y_min-2)] -
                     mom_flux[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)]) /
                    node_mass_post[FTNREF2D(j  ,k  ,x_max+5,x_min-2,y_min-2)];
            }
        }
    }

    STOP_PROFILING(__func__);
}
