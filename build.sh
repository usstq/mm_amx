
CXX=icx
if ! which icx > /dev/null; then
    echo "use g++ instead of intel compiler (AMX intrinsic maybe missing)"
    CXX=g++
fi
source=$1

if ! test -f "${source}"; then
    echo "cannot find input source cpp file: '$source'"
    exit 1
fi

#https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# target=a`git rev-parse --short HEAD`.out
target=a.out

# g++ ./test.cpp -O2 -lpthread -march=native -lstdc++

MARCH_OPTS="-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store"
MARCH_OPTS=""
COMMON_OPTS="-DENABLE_NUMA -I$SCRIPT_DIR/include -lpthread -march=native -std=c++14 -lstdc++ -lnuma -fopenmp $MARCH_OPTS"

$CXX $source -O2 $COMMON_OPTS -S -masm=intel -fverbose-asm  -o _main.s &&
cat _main.s | c++filt > main.s &&
$CXX $source -O2 $COMMON_OPTS -o $target &&
$CXX $source -O0 $COMMON_OPTS -g -o debug.out &&
echo $target is generated &&
echo main.s is generated &&
echo debug.out is generated &&
echo ======== test begin========== &&
echo numactl --localalloc -C 0-5 ./$target &&
numactl --localalloc -C 0-5 ./$target
