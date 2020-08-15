make PARALLEL=1 CFLAGS+="-fPIC"
OMP=0
gcc -shared -Xpreprocessor  -fopenmp -o bart.so src/bart.o -Wl, lib/lib*.a -Wl, -Wl, -Bstatic -Wl, -L/opt/local/lib/lapack -Bdynamic -llapacke -lblas -llapack -lblas -lfftw3f -lfftw3f_threads -L/usr/local/Cellar/gcc/10.2.0/lib/gcc/10/ -lquadmath -lz -L/opt/local/lib -lpng -lm -luuid

# Lapack headers installed in /opt/local/include/lapack
# centralize mac os package management