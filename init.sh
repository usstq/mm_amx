
export CXX=icpx

if ! which icpx > /dev/null; then
# source oneapi & try again
source /opt/intel/oneapi/setvars.sh
if ! which icpx > /dev/null; then
    echo "use g++ instead of intel compiler (AMX intrinsic maybe missing)"
    export CXX=g++
fi
fi

#https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# g++ ./test.cpp -O2 -lpthread -march=native -lstdc++
export LD_LIBRARY_PATH=thirdparty/oneDNN/build/install/lib64:$LD_LIBRARY_PATH

# target=a`git rev-parse --short HEAD`.out
function build() {
    source=$1

    if ! test -f "${source}"; then
        echo "cannot find input source cpp file: '$source'"
        return
    fi
    target=a.out

    MARCH_OPTS="-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store"
    MARCH_OPTS=""
    COMMON_OPTS="-DENABLE_NUMA -I$SCRIPT_DIR/include -Ithirdparty/oneDNN/build/install/include -Ithirdparty/xbyak/xbyak -Lthirdparty/oneDNN/build/install/lib64 -lpthread -ldnnl -march=native -std=c++14 -lstdc++ -lnuma -fopenmp $MARCH_OPTS"

    $CXX $source -O2 $COMMON_OPTS -S -masm=intel -fverbose-asm  -o _main.s &&
    cat _main.s | c++filt > main.s &&
    $CXX $source -O2 $COMMON_OPTS -o $target &&
    $CXX $source -O0 $COMMON_OPTS -g -DJIT_DEBUG -o debug.out &&
    echo $target is generated &&
    echo main.s is generated &&
    echo debug.out is generated &&
    echo ======== test begin========== &&
    echo numactl --localalloc -C 0-5 ./$target &&
    CLFLUSH=1 numactl -N1 --localalloc -C56 ./$target
}

function run1() {
    CLFLUSH=1 numactl --localalloc -C56 $1
}
