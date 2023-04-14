if ! which icx > /dev/null; then
echo "please execute following command to initialize IntelCompiler env"
echo source ~/intel/oneapi/setvars.sh
exit 1
fi


source=$1

if ! test -f "${source}"; then
    echo "cannot find input source cpp file: '$source'"
    exit 1
fi

# target=a`git rev-parse --short HEAD`.out
target=a.out

# g++ ./test.cpp -O2 -lpthread -march=native -lstdc++

COMMON_OPTS="-DENABLE_NUMA -I../include -lpthread -march=native -lstdc++ -lnuma -qopenmp"

icx $source -O2 $COMMON_OPTS -S -masm=intel -fverbose-asm  -o _main.s &&
cat _main.s | c++filt > main.s &&
icx $source -O2 $COMMON_OPTS -o $target &&
icx $source -O0 $COMMON_OPTS -g -o debug.out &&
echo $target is generated &&
echo main.s is generated &&
echo debug.out is generated &&
echo ======== test begin========== &&
echo numactl --localalloc -C 0-55 ./$target &&
numactl --localalloc -C 0-55 ./$target
