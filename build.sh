target=a`git rev-parse --short HEAD`.out

echo source ~/intel/oneapi/setvars.sh

# g++ ./test.cpp -O2 -lpthread -march=native -lstdc++

icx ./main.cpp -O2 -I./include -lpthread -march=native -lstdc++ -S -masm=intel -fverbose-asm  -o _main.s &&
cat _main.s | c++filt > main.s &&
icx ./main.cpp -O2 -I./include -lpthread -march=native -lstdc++ -o $target &&
icx ./main.cpp -O0 -g -I./include -lpthread -march=native -lstdc++ -o debug.out &&
./$target &&
echo main.s is generated &&
echo debug.out is generated &&
echo $target is generated