target=a_`git rev-parse --short HEAD`.out

echo source ~/intel/oneapi/setvars.sh

icx ./mm_amx_bf16.cpp -O2 -lpthread -march=native -lstdc++ -S -masm=intel -fverbose-asm  -o _asm.s &&
cat _asm.s | c++filt > asm.s &&
echo asm.s is generated &&
icx ./mm_amx_bf16.cpp -O2 -lpthread -march=native -lstdc++ -o $target &&
echo $target is generated

./$target