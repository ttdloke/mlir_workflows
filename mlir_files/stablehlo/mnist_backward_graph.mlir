module @jit_update attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<512x784xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<512xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<512x512xf32> {mhlo.layout_mode = "default"}, %arg3: tensor<512xf32> {mhlo.layout_mode = "default"}, %arg4: tensor<10x512xf32> {mhlo.layout_mode = "default"}, %arg5: tensor<10xf32> {mhlo.layout_mode = "default"}, %arg6: tensor<96x784xui8> {mhlo.layout_mode = "default"}, %arg7: tensor<96x10xf32> {mhlo.layout_mode = "default"}) -> (tensor<512x784xf32> {jax.result_info = "[0][0]", mhlo.layout_mode = "default"}, tensor<512xf32> {jax.result_info = "[0][1]", mhlo.layout_mode = "default"}, tensor<512x512xf32> {jax.result_info = "[1][0]", mhlo.layout_mode = "default"}, tensor<512xf32> {jax.result_info = "[1][1]", mhlo.layout_mode = "default"}, tensor<10x512xf32> {jax.result_info = "[2][0]", mhlo.layout_mode = "default"}, tensor<10xf32> {jax.result_info = "[2][1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : tensor<512x784xf32>
    %1 = stablehlo.convert %arg6 : (tensor<96x784xui8>) -> tensor<96x784xf32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x784xf32>, tensor<96x784xf32>) -> tensor<512x96xf32>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<512x96xf32>) -> tensor<96x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<96x512xf32>
    %6 = stablehlo.add %3, %5 : tensor<96x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<96x512xf32>
    %8 = stablehlo.maximum %7, %6 : tensor<96x512xf32>
    %9 = stablehlo.compare  EQ, %6, %8,  FLOAT : (tensor<96x512xf32>, tensor<96x512xf32>) -> tensor<96x512xi1>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<96x512xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<96x512xf32>
    %12 = stablehlo.select %9, %10, %11 : tensor<96x512xi1>, tensor<96x512xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<96x512xf32>
    %14 = stablehlo.compare  EQ, %13, %8,  FLOAT : (tensor<96x512xf32>, tensor<96x512xf32>) -> tensor<96x512xi1>
    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<96x512xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<96x512xf32>
    %17 = stablehlo.select %14, %15, %16 : tensor<96x512xi1>, tensor<96x512xf32>
    %18 = stablehlo.divide %12, %17 : tensor<96x512xf32>
    %19 = stablehlo.dot_general %arg2, %8, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x512xf32>, tensor<96x512xf32>) -> tensor<512x96xf32>
    %20 = stablehlo.transpose %19, dims = [1, 0] : (tensor<512x96xf32>) -> tensor<96x512xf32>
    %21 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<96x512xf32>
    %23 = stablehlo.add %20, %22 : tensor<96x512xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<96x512xf32>
    %25 = stablehlo.maximum %24, %23 : tensor<96x512xf32>
    %26 = stablehlo.compare  EQ, %23, %25,  FLOAT : (tensor<96x512xf32>, tensor<96x512xf32>) -> tensor<96x512xi1>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<96x512xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %28 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<96x512xf32>
    %29 = stablehlo.select %26, %27, %28 : tensor<96x512xi1>, tensor<96x512xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<96x512xf32>
    %31 = stablehlo.compare  EQ, %30, %25,  FLOAT : (tensor<96x512xf32>, tensor<96x512xf32>) -> tensor<96x512xi1>
    %cst_9 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<96x512xf32>
    %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<96x512xf32>
    %34 = stablehlo.select %31, %32, %33 : tensor<96x512xi1>, tensor<96x512xf32>
    %35 = stablehlo.divide %29, %34 : tensor<96x512xf32>
    %36 = stablehlo.dot_general %arg4, %25, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<10x512xf32>, tensor<96x512xf32>) -> tensor<10x96xf32>
    %37 = stablehlo.transpose %36, dims = [1, 0] : (tensor<10x96xf32>) -> tensor<96x10xf32>
    %38 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<10xf32>) -> tensor<1x10xf32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1] : (tensor<1x10xf32>) -> tensor<96x10xf32>
    %40 = stablehlo.add %37, %39 : tensor<96x10xf32>
    %cst_11 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %41 = stablehlo.reduce(%40 init: %cst_11) applies stablehlo.maximum across dimensions = [1] : (tensor<96x10xf32>, tensor<f32>) -> tensor<96xf32>
    %cst_12 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %43 = stablehlo.maximum %42, %41 : tensor<96xf32>
    %44 = stablehlo.is_finite %43 : (tensor<96xf32>) -> tensor<96xi1>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %46 = stablehlo.select %44, %43, %45 : tensor<96xi1>, tensor<96xf32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [0] : (tensor<96xf32>) -> tensor<96x1xf32>
    %48 = stablehlo.broadcast_in_dim %47, dims = [0, 1] : (tensor<96x1xf32>) -> tensor<96x10xf32>
    %49 = stablehlo.subtract %40, %48 : tensor<96x10xf32>
    %50 = stablehlo.exponential %49 : tensor<96x10xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %51 = stablehlo.reduce(%50 init: %cst_14) applies stablehlo.add across dimensions = [1] : (tensor<96x10xf32>, tensor<f32>) -> tensor<96xf32>
    %52 = stablehlo.abs %51 : tensor<96xf32>
    %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %54 = stablehlo.compare  GE, %51, %53,  FLOAT : (tensor<96xf32>, tensor<96xf32>) -> tensor<96xi1>
    %cst_16 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %55 = stablehlo.negate %cst_16 : tensor<f32>
    %cst_17 = stablehlo.constant dense<9.600000e+02> : tensor<f32>
    %56 = stablehlo.divide %55, %cst_17 : tensor<f32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [] : (tensor<f32>) -> tensor<96x10xf32>
    %58 = stablehlo.multiply %57, %arg7 : tensor<96x10xf32>
    %59 = stablehlo.negate %58 : tensor<96x10xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %60 = stablehlo.reduce(%59 init: %cst_18) applies stablehlo.add across dimensions = [1] : (tensor<96x10xf32>, tensor<f32>) -> tensor<96xf32>
    %61 = stablehlo.reshape %60 : (tensor<96xf32>) -> tensor<96x1xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %62 = stablehlo.reduce(%61 init: %cst_19) applies stablehlo.add across dimensions = [1] : (tensor<96x1xf32>, tensor<f32>) -> tensor<96xf32>
    %63 = stablehlo.divide %62, %52 : tensor<96xf32>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %64 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %65 = stablehlo.select %54, %64, %63 : tensor<96xi1>, tensor<96xf32>
    %66 = stablehlo.select %54, %63, %64 : tensor<96xi1>, tensor<96xf32>
    %67 = stablehlo.negate %65 : tensor<96xf32>
    %68 = stablehlo.add %66, %67 : tensor<96xf32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [0] : (tensor<96xf32>) -> tensor<96x10xf32>
    %70 = stablehlo.multiply %69, %50 : tensor<96x10xf32>
    %71 = stablehlo.add %58, %70 : tensor<96x10xf32>
    %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %72 = stablehlo.reduce(%71 init: %cst_21) applies stablehlo.add across dimensions = [0] : (tensor<96x10xf32>, tensor<f32>) -> tensor<10xf32>
    %73 = stablehlo.reshape %72 : (tensor<10xf32>) -> tensor<1x10xf32>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %74 = stablehlo.reduce(%73 init: %cst_22) applies stablehlo.add across dimensions = [0] : (tensor<1x10xf32>, tensor<f32>) -> tensor<10xf32>
    %75 = stablehlo.transpose %71, dims = [1, 0] : (tensor<96x10xf32>) -> tensor<10x96xf32>
    %76 = stablehlo.dot_general %75, %arg4, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<10x96xf32>, tensor<10x512xf32>) -> tensor<96x512xf32>
    %77 = stablehlo.dot_general %75, %25, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<10x96xf32>, tensor<96x512xf32>) -> tensor<10x512xf32>
    %78 = stablehlo.multiply %76, %35 : tensor<96x512xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %79 = stablehlo.reduce(%78 init: %cst_23) applies stablehlo.add across dimensions = [0] : (tensor<96x512xf32>, tensor<f32>) -> tensor<512xf32>
    %80 = stablehlo.reshape %79 : (tensor<512xf32>) -> tensor<1x512xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %81 = stablehlo.reduce(%80 init: %cst_24) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %82 = stablehlo.transpose %78, dims = [1, 0] : (tensor<96x512xf32>) -> tensor<512x96xf32>
    %83 = stablehlo.dot_general %82, %arg2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512x96xf32>, tensor<512x512xf32>) -> tensor<96x512xf32>
    %84 = stablehlo.dot_general %82, %8, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512x96xf32>, tensor<96x512xf32>) -> tensor<512x512xf32>
    %85 = stablehlo.multiply %83, %18 : tensor<96x512xf32>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %86 = stablehlo.reduce(%85 init: %cst_25) applies stablehlo.add across dimensions = [0] : (tensor<96x512xf32>, tensor<f32>) -> tensor<512xf32>
    %87 = stablehlo.reshape %86 : (tensor<512xf32>) -> tensor<1x512xf32>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %88 = stablehlo.reduce(%87 init: %cst_26) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %89 = stablehlo.transpose %85, dims = [1, 0] : (tensor<96x512xf32>) -> tensor<512x96xf32>
    %90 = stablehlo.convert %89 : tensor<512x96xf32>
    %91 = stablehlo.convert %arg6 : (tensor<96x784xui8>) -> tensor<96x784xf32>
    %92 = stablehlo.dot_general %90, %91, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512x96xf32>, tensor<96x784xf32>) -> tensor<512x784xf32>
    %cst_27 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %93 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %94 = stablehlo.multiply %93, %92 : tensor<512x784xf32>
    %95 = stablehlo.subtract %arg0, %94 : tensor<512x784xf32>
    %cst_28 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %96 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %97 = stablehlo.multiply %96, %88 : tensor<512xf32>
    %98 = stablehlo.subtract %arg1, %97 : tensor<512xf32>
    %cst_29 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %99 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %100 = stablehlo.multiply %99, %84 : tensor<512x512xf32>
    %101 = stablehlo.subtract %arg2, %100 : tensor<512x512xf32>
    %cst_30 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %102 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %103 = stablehlo.multiply %102, %81 : tensor<512xf32>
    %104 = stablehlo.subtract %arg3, %103 : tensor<512xf32>
    %cst_31 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %105 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<10x512xf32>
    %106 = stablehlo.multiply %105, %77 : tensor<10x512xf32>
    %107 = stablehlo.subtract %arg4, %106 : tensor<10x512xf32>
    %cst_32 = stablehlo.constant dense<0.00999999977> : tensor<f32>
    %108 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %109 = stablehlo.multiply %108, %74 : tensor<10xf32>
    %110 = stablehlo.subtract %arg5, %109 : tensor<10xf32>
    return %95, %98, %101, %104, %107, %110 : tensor<512x784xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<10x512xf32>, tensor<10xf32>
  }
}