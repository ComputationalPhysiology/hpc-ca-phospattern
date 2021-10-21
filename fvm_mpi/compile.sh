MAKEFILE="Makefile.gcc"
#MAKEFILE="Makefile.pluto"

MYPATH="/home/chad/run_diff"
#MYPATH="/work/jh180024/k27003/run_diff"

make clean
make -j -f $MAKEFILE
cp bin/calcium_sparks $MYPATH/calcium_sparks_LOT

make clean
make -j -f $MAKEFILE MANUAL=1 
cp bin/calcium_sparks $MYPATH/calcium_sparks_LOT_manual1

make clean
make -j -f $MAKEFILE MANUAL=2
cp bin/calcium_sparks $MYPATH/calcium_sparks_LOT_manual2

make clean
make -j -f $MAKEFILE ALPHA=1
cp bin/calcium_sparks $MYPATH/calcium_sparks_alpha

make clean
make -j -f $MAKEFILE ALPHA=1 MANUAL=1
cp bin/calcium_sparks $MYPATH/calcium_sparks_alpha_manual1

make clean
make -j -f $MAKEFILE MANUAL=2 OMP=1
cp bin/calcium_sparks $MYPATH/calcium_sparks_LOT_manual2_omp

make clean
make -j -f $MAKEFILE NOVEC=1
cp bin/calcium_sparks $MYPATH/calcium_sparks_LOT_novector

make clean
make -j -f $MAKEFILE ALPHA=1 NOVEC=1
cp bin/calcium_sparks $MYPATH/calcium_sparks_alpha_novector

make clean
make -j -f $MAKEFILE NOVEC=1 SIMD=!
cp bin/calcium_sparks $MYPATH/calcium_sparks_LOT_simd

make clean
make -j -f $MAKEFILE ALPHA=1 SIMD=1
cp bin/calcium_sparks $MYPATH/calcium_sparks_alpha_simd

make clean
make -j -f $MAKEFILE MANUAL=1 GATHERONLY=1
cp bin/calcium_sparks $MYPATH/calcium_sparks_gather
