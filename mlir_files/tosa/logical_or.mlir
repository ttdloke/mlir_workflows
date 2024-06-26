module attributes {torch.debug_module_name = "Model"} {
  func.func @forward(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<256x256xi1> {
    %0 = tosa.cast %arg0 : (tensor<256x256xf32>) -> tensor<256x256xi1>
    %1 = tosa.cast %arg1 : (tensor<256x256xf32>) -> tensor<256x256xi1>
    %2 = tosa.logical_or %0, %1 : (tensor<256x256xi1>, tensor<256x256xi1>) -> tensor<256x256xi1>
    return %2 : tensor<256x256xi1>
  }
}

