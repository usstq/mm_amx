target=a`git rev-parse --short HEAD`.out

echo source ~/intel/oneapi/setvars.sh

# g++ ./test.cpp -O2 -lpthread -march=native -lstdc++

COMMON_OPTS="-DENABLE_NUMA -I../include -lpthread -march=native -lstdc++ -lnuma -qopenmp"

icx ./main.cpp -O2 $COMMON_OPTS -S -masm=intel -fverbose-asm  -o _main.s &&
cat _main.s | c++filt > main.s &&
icx ./main.cpp -O2 $COMMON_OPTS -o $target &&
icx ./main.cpp -O0 $COMMON_OPTS -g -o debug.out &&
echo $target is generated &&
echo main.s is generated &&
echo debug.out is generated &&
echo ======== test begin========== &&
echo numactl --localalloc -C 0-55 ./$target &&
numactl --localalloc -C 0-55 ./$target
