module attributes {torch.debug_module_name = "Model"} {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<128x64xf32>) -> tensor<64x64xf32> {
    %0 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1 = tosa.reshape %arg0 {new_shape = array<i64: 1, 64, 128>} : (tensor<64x128xf32>) -> tensor<1x64x128xf32>
    %2 = tosa.reshape %arg1 {new_shape = array<i64: 1, 128, 64>} : (tensor<128x64xf32>) -> tensor<1x128x64xf32>
    %3 = tosa.matmul %1, %2 : (tensor<1x64x128xf32>, tensor<1x128x64xf32>) -> tensor<1x64x64xf32>
    %4 = tosa.reshape %3 {new_shape = array<i64: 64, 64>} : (tensor<1x64x64xf32>) -> tensor<64x64xf32>
    %5 = tosa.transpose %4, %0 : (tensor<64x64xf32>, tensor<2xi32>) -> tensor<64x64xf32>
    return %5 : tensor<64x64xf32>
  }
}

