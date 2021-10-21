OMP=-fopenmp
#unset OMP
#g++ -c -O3 -march=nehalem kernel_generic.cpp ${OMP} -fno-exceptions
#g++ -c -O3 -march=sandybridge kernel_generic.cpp ${OMP} -fno-exceptions
#g++ -c -O3 -march=skylake kernel_generic.cpp ${OMP} -fno-exceptions
g++ -c -O3 -march=skylake kernel_generic.cpp -fno-exceptions
#g++ -c -O3 -march=skylake-avx512 kernel_generic.cpp ${OMP} -fno-exceptions
#g++ -c -O3 -march=knl kernel_generic.cpp ${OMP} -fno-exceptions

#gcc -O2 ${OMP} -DVERSION=1 kernel.c kernel_generic.o -o kernel_LUT_auto
gcc -O2 ${OMP} -DVERSION=2 kernel.c kernel_generic.o -o kernel_LUT_simd
gcc -O2 ${OMP} -DVERSION=3 kernel.c kernel_generic.o -o kernel_CA
gcc -O2 ${OMP} -DVERSION=4 kernel.c kernel_generic.o -o kernel_block

#echo "auto"
#time ./kernel_LUT_auto
echo "simd"
./kernel_LUT_simd
echo "CA"
./kernel_CA
echo "block"
./kernel_block

#echo "auto"
#perf stat -d ./kernel_LUT_auto
#echo "simd"
#perf stat -d ./kernel_LUT_simd
#echo "CA"
#perf stat -d ./kernel_CA
#echo "block"
#perf stat -d ./kernel_block

