module attributes {torch.debug_module_name = "Model"} {
  func.func @forward(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = tosa.floor %arg0 : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1 = tosa.floor %arg1 : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2 = tosa.reciprocal %1 : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %3 = tosa.mul %0, %2 {shift = 0 : i8} : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %4 = tosa.ceil %3 : (tensor<256x256xf32>) -> tensor<256x256xf32>
    return %4 : tensor<256x256xf32>
  }
}