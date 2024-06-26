module attributes {torch.debug_module_name = "Model"} {
  func.func @forward(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = tosa.sigmoid %arg0 : (tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}

