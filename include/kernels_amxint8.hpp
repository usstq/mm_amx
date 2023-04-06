#pragma once

namespace amx_int8 {

//
//   C    =    A   *  B
//  (MxN)    (MxK)  (KxN)   
//
// M: number of tokens
// K: channels (feature size)
// N: output channels
//
// X*W is performed in INT8 : C_int32_16x16 = A_int8_16x64 * B_int8_64x16 (re-layout as 16x16x4 Ab4a)
// and the result is undergone a post-process:
//    DeQquantize => Activation => [Quantize]
//
// Using SmoothQuant (https://arxiv.org/abs/2211.10438)
//
//    A is quantized in per-input channel(K) style
//    B is quantized in per-input channel(K) style
//    DeQquantize is done with per-tensor scale
//
//    Quantize, if required, is done with per-output channel(N) style
//    if next OP is also using SmoothQuant.
//
// we will need:
//
//   calibration: collecte min/max value on each channels 
//
//
//

};
