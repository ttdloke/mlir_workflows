module attributes {torch.debug_module_name = "Eltwise"} {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<128x64xf32>) -> tensor<64x64xf32> {
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 64, 128>} : (tensor<64x128xf32>) -> tensor<1x64x128xf32>
    %1 = tosa.reshape %arg1 {new_shape = array<i64: 1, 128, 64>} : (tensor<128x64xf32>) -> tensor<1x128x64xf32>
    %2 = tosa.matmul %0, %1 : (tensor<1x64x128xf32>, tensor<1x128x64xf32>) -> tensor<1x64x64xf32>
    %3 = tosa.reshape %2 {new_shape = array<i64: 64, 64>} : (tensor<1x64x64xf32>) -> tensor<64x64xf32>
    return %3 : tensor<64x64xf32>
  }
}