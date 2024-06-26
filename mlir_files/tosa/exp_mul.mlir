module attributes {torch.debug_module_name = "Model"} {
  func.func @forward(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = tosa.exp %arg0 : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1 = tosa.exp %arg1 : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2 = tosa.mul %0, %1 {shift = 0 : i8} : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    return %2 : tensor<256x256xf32>
  }
}

