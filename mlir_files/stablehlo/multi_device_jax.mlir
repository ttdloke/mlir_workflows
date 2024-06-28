module @jit_dot attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 3 : i32} {
  func.func public @main(%arg0: tensor<3x2x2xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<3x2x2xf32> {mhlo.layout_mode = "default"}) -> (tensor<3x2x2xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.constant dense<0> : tensor<ui32>
    %1 = stablehlo.constant dense<1> : tensor<ui32>
    %2 = stablehlo.constant dense<3> : tensor<ui32>
    %3 = stablehlo.replica_id : tensor<ui32>
    %4 = stablehlo.divide %3, %1 : tensor<ui32>
    %5 = stablehlo.remainder %4, %2 : tensor<ui32>
    %6 = stablehlo.dynamic_slice %arg0, %5, %0, %0, sizes = [1, 2, 2] : (tensor<3x2x2xf32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x2x2xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x2x2xf32>) -> tensor<2x2xf32>
    %8 = stablehlo.constant dense<0> : tensor<ui32>
    %9 = stablehlo.constant dense<1> : tensor<ui32>
    %10 = stablehlo.constant dense<3> : tensor<ui32>
    %11 = stablehlo.replica_id : tensor<ui32>
    %12 = stablehlo.divide %11, %9 : tensor<ui32>
    %13 = stablehlo.remainder %12, %10 : tensor<ui32>
    %14 = stablehlo.dynamic_slice %arg1, %13, %8, %8, sizes = [1, 2, 2] : (tensor<3x2x2xf32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x2x2xf32>
    %15 = stablehlo.reshape %14 : (tensor<1x2x2xf32>) -> tensor<2x2xf32>
    %16 = stablehlo.dot_general %7, %15, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<3x2x2xf32>
    %19 = stablehlo.constant dense<0> : tensor<ui32>
    %20 = stablehlo.constant dense<1> : tensor<ui32>
    %21 = stablehlo.constant dense<3> : tensor<ui32>
    %22 = stablehlo.replica_id : tensor<ui32>
    %23 = stablehlo.divide %22, %20 : tensor<ui32>
    %24 = stablehlo.remainder %23, %21 : tensor<ui32>
    %25 = stablehlo.broadcast %16, sizes = [1] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %26 = stablehlo.dynamic_update_slice %18, %25, %24, %19, %19 : (tensor<3x2x2xf32>, tensor<1x2x2xf32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<3x2x2xf32>
    %27 = "stablehlo.cross-replica-sum"(%26) {replica_groups = dense<[[0, 1, 2]]> : tensor<1x3xi64>} : (tensor<3x2x2xf32>) -> tensor<3x2x2xf32>
    return %27 : tensor<3x2x2xf32>
  }
}