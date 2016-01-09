set title 'Fluid Velocity'
set xlabel 'cell # along x-dimension'
set ylabel 'cell # along y-dimension'
set size ratio -1
set autoscale fix

set terminal png
set output 'vis.png'

plot 'clover.dat' using 1:2:3 with image
