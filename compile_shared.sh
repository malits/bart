make PARALLEL=1 CFLAGS+="-fPIC" OMP=0
gcc -shared -fopenmp -o bart.so src/bart.o -Wl,--whole-archive lib/lib*.a -Wl,--no-whole-archive -Wl,-Bstatic -Wl,-Bdynamic -llapacke -lblas -llapack -lblas -lquadmath -lz -lpng -lfftw3f -lfftw3f_threads -lm -luuid

mv bart.so python/bart/

# Lapack headers installed in /opt/local/include/lapack
# centralize mac os package management
