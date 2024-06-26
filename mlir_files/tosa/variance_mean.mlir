module attributes {torch.debug_module_name = "Model"} {
  func.func @forward(%arg0: tensor<256x128x4x8x1x16xf32>) -> (tensor<128x4x1xf32>, tensor<128x4x1xf32>) {
    %0 = "tosa.const"() <{value = dense<3.0517578125E-5> : tensor<1x1x1x1x1x1xf64>}> : () -> tensor<1x1x1x1x1x1xf64>
    %1 = "tosa.const"() <{value = dense<3.0518509475997192E-5> : tensor<1x1x1xf64>}> : () -> tensor<1x1x1xf64>
    %2 = "tosa.const"() <{value = dense<3.05175781E-5> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %3 = tosa.cast %arg0 : (tensor<256x128x4x8x1x16xf32>) -> tensor<256x128x4x8x1x16xf64>
    %4 = tosa.reduce_sum %3 {axis = 0 : i32} : (tensor<256x128x4x8x1x16xf64>) -> tensor<1x128x4x8x1x16xf64>
    %5 = tosa.reduce_sum %4 {axis = 3 : i32} : (tensor<1x128x4x8x1x16xf64>) -> tensor<1x128x4x1x1x16xf64>
    %6 = tosa.reduce_sum %5 {axis = 5 : i32} : (tensor<1x128x4x1x1x16xf64>) -> tensor<1x128x4x1x1x1xf64>
    %7 = tosa.mul %6, %0 {shift = 0 : i8} : (tensor<1x128x4x1x1x1xf64>, tensor<1x1x1x1x1x1xf64>) -> tensor<1x128x4x1x1x1xf64>
    %8 = tosa.sub %3, %7 : (tensor<256x128x4x8x1x16xf64>, tensor<1x128x4x1x1x1xf64>) -> tensor<256x128x4x8x1x16xf64>
    %9 = tosa.mul %8, %8 {shift = 0 : i8} : (tensor<256x128x4x8x1x16xf64>, tensor<256x128x4x8x1x16xf64>) -> tensor<256x128x4x8x1x16xf64>
    %10 = tosa.reduce_sum %9 {axis = 0 : i32} : (tensor<256x128x4x8x1x16xf64>) -> tensor<1x128x4x8x1x16xf64>
    %11 = tosa.reduce_sum %10 {axis = 3 : i32} : (tensor<1x128x4x8x1x16xf64>) -> tensor<1x128x4x1x1x16xf64>
    %12 = tosa.reduce_sum %11 {axis = 5 : i32} : (tensor<1x128x4x1x1x16xf64>) -> tensor<1x128x4x1x1x1xf64>
    %13 = tosa.reshape %12 {new_shape = array<i64: 128, 4, 1>} : (tensor<1x128x4x1x1x1xf64>) -> tensor<128x4x1xf64>
    %14 = tosa.mul %13, %1 {shift = 0 : i8} : (tensor<128x4x1xf64>, tensor<1x1x1xf64>) -> tensor<128x4x1xf64>
    %15 = tosa.cast %14 : (tensor<128x4x1xf64>) -> tensor<128x4x1xf32>
    %16 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<256x128x4x8x1x16xf32>) -> tensor<1x128x4x8x1x16xf32>
    %17 = tosa.reduce_sum %16 {axis = 3 : i32} : (tensor<1x128x4x8x1x16xf32>) -> tensor<1x128x4x1x1x16xf32>
    %18 = tosa.reduce_sum %17 {axis = 5 : i32} : (tensor<1x128x4x1x1x16xf32>) -> tensor<1x128x4x1x1x1xf32>
    %19 = tosa.reshape %18 {new_shape = array<i64: 128, 4, 1>} : (tensor<1x128x4x1x1x1xf32>) -> tensor<128x4x1xf32>
    %20 = tosa.mul %19, %2 {shift = 0 : i8} : (tensor<128x4x1xf32>, tensor<1x1x1xf32>) -> tensor<128x4x1xf32>
    return %15, %20 : tensor<128x4x1xf32>, tensor<128x4x1xf32>
  }
}

