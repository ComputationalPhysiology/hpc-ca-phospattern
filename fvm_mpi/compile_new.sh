MAKEFILE="Makefile.gcc"
#MAKEFILE="Makefile.pluto"

MYPATH="/home/chad/run_diff"
#MYPATH="/work/jh180024/k27003/run_diff"

# make clean
# make -j -f $MAKEFILE ARCH=4
# cp bin/calcium_sparks $MYPATH/calcium_sparks_LUT_AVX512

# make clean
# make -j -f $MAKEFILE ARCH=3
# cp bin/calcium_sparks $MYPATH/calcium_sparks_LUT_AVX2

# make clean
# make -j -f $MAKEFILE ARCH=2
# cp bin/calcium_sparks $MYPATH/calcium_sparks_LUT_AVX

# make clean
# make -j -f $MAKEFILE ARCH=1
# cp bin/calcium_sparks $MYPATH/calcium_sparks_LUT_SSE4
mpicc.mpich 
make clean
make -j -f $MAKEFILE ARCH=4 GATHERONLY=1 CC=mpicc.mpich CXX=g++
cp bin/calcium_sparks $MYPATH/calcium_sparks_LUT_gatheronly_AVX512

make clean
make -j -f $MAKEFILE ARCH=3 GATHERONLY=1 CC=mpicc.mpich CXX=g++
cp bin/calcium_sparks $MYPATH/calcium_sparks_LUT_gatheronly_AVX2

make clean
make -j -f $MAKEFILE ARCH=2 GATHERONLY=1 CC=mpicc.mpich CXX=g++
cp bin/calcium_sparks $MYPATH/calcium_sparks_LUT_gatheronly_AVX

make clean
make -j -f $MAKEFILE ARCH=1 GATHERONLY=1 CC=mpicc.mpich CXX=g++
cp bin/calcium_sparks $MYPATH/calcium_sparks_LUT_gatheronly_SSE4

make clean
make -j -f $MAKEFILE ARCH=4 ALPHA=1 CC=mpicc.mpich CXX=g++
cp bin/calcium_sparks $MYPATH/calcium_sparks_alpha_AVX512

make clean
make -j -f $MAKEFILE ARCH=3 ALPHA=1 CC=mpicc.mpich CXX=g++
cp bin/calcium_sparks $MYPATH/calcium_sparks_alpha_AVX2

make clean
make -j -f $MAKEFILE ARCH=2 ALPHA=1 CC=mpicc.mpich CXX=g++
cp bin/calcium_sparks $MYPATH/calcium_sparks_alpha_AVX

make clean
make -j -f $MAKEFILE ARCH=1 ALPHA=1 CC=mpicc.mpich CXX=g++
cp bin/calcium_sparks $MYPATH/calcium_sparks_alpha_SSE4

