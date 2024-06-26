module @jit__unnamed_wrapped_function_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x3x224x224xf32> {mhlo.layout_mode = "default"}) -> (tensor<1x512x7x7xf32> {mhlo.layout_mode = "default"}, tensor<1x512x1x1xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense_resource<__elided__> : tensor<7x7x3x64xf32>
    %cst_0 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_1 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_2 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_3 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_4 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x64x64xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<64xf32>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<64xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_9 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x64x64xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<64xf32>
    %cst_12 = stablehlo.constant dense<1.000000e+00> : tensor<64xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_14 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x64x64xf32>
    %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_16 = stablehlo.constant dense<1.000000e+00> : tensor<64xf32>
    %cst_17 = stablehlo.constant dense<1.000000e+00> : tensor<64xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_19 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x64x64xf32>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_21 = stablehlo.constant dense<1.000000e+00> : tensor<64xf32>
    %cst_22 = stablehlo.constant dense<1.000000e+00> : tensor<64xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_24 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x64x128xf32>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_26 = stablehlo.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_27 = stablehlo.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_28 = stablehlo.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_29 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x128x128xf32>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_31 = stablehlo.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_32 = stablehlo.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_33 = stablehlo.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_34 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x64x128xf32>
    %cst_35 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_36 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_37 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_38 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_39 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x128x128xf32>
    %cst_40 = stablehlo.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_41 = stablehlo.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_42 = stablehlo.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_43 = stablehlo.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_44 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x128x128xf32>
    %cst_45 = stablehlo.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_46 = stablehlo.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_47 = stablehlo.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_48 = stablehlo.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_49 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x128x256xf32>
    %cst_50 = stablehlo.constant dense<0.000000e+00> : tensor<256xf32>
    %cst_51 = stablehlo.constant dense<1.000000e+00> : tensor<256xf32>
    %cst_52 = stablehlo.constant dense<1.000000e+00> : tensor<256xf32>
    %cst_53 = stablehlo.constant dense<0.000000e+00> : tensor<256xf32>
    %cst_54 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x256xf32>
    %cst_55 = stablehlo.constant dense<0.000000e+00> : tensor<256xf32>
    %cst_56 = stablehlo.constant dense<1.000000e+00> : tensor<256xf32>
    %cst_57 = stablehlo.constant dense<1.000000e+00> : tensor<256xf32>
    %cst_58 = stablehlo.constant dense<0.000000e+00> : tensor<256xf32>
    %cst_59 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x128x256xf32>
    %cst_60 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_61 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_62 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_63 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_64 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x256xf32>
    %cst_65 = stablehlo.constant dense<0.000000e+00> : tensor<256xf32>
    %cst_66 = stablehlo.constant dense<1.000000e+00> : tensor<256xf32>
    %cst_67 = stablehlo.constant dense<1.000000e+00> : tensor<256xf32>
    %cst_68 = stablehlo.constant dense<0.000000e+00> : tensor<256xf32>
    %cst_69 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x256xf32>
    %cst_70 = stablehlo.constant dense<0.000000e+00> : tensor<256xf32>
    %cst_71 = stablehlo.constant dense<1.000000e+00> : tensor<256xf32>
    %cst_72 = stablehlo.constant dense<1.000000e+00> : tensor<256xf32>
    %cst_73 = stablehlo.constant dense<0.000000e+00> : tensor<256xf32>
    %cst_74 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x512xf32>
    %cst_75 = stablehlo.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_76 = stablehlo.constant dense<1.000000e+00> : tensor<512xf32>
    %cst_77 = stablehlo.constant dense<1.000000e+00> : tensor<512xf32>
    %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_79 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x512x512xf32>
    %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_81 = stablehlo.constant dense<1.000000e+00> : tensor<512xf32>
    %cst_82 = stablehlo.constant dense<1.000000e+00> : tensor<512xf32>
    %cst_83 = stablehlo.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_84 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x512xf32>
    %cst_85 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_86 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_87 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_88 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_89 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x512x512xf32>
    %cst_90 = stablehlo.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_91 = stablehlo.constant dense<1.000000e+00> : tensor<512xf32>
    %cst_92 = stablehlo.constant dense<1.000000e+00> : tensor<512xf32>
    %cst_93 = stablehlo.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_94 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x512x512xf32>
    %cst_95 = stablehlo.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_96 = stablehlo.constant dense<1.000000e+00> : tensor<512xf32>
    %cst_97 = stablehlo.constant dense<1.000000e+00> : tensor<512xf32>
    %cst_98 = stablehlo.constant dense<0.000000e+00> : tensor<512xf32>
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
    %1 = stablehlo.convolution(%0, %cst) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x224x224x3xf32>, tensor<7x7x3x64xf32>) -> tensor<1x112x112x64xf32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<1x112x112x64xf32>
    %cst_99 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_99, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %7 = stablehlo.add %3, %6 : tensor<1x1x1x64xf32>
    %8 = stablehlo.rsqrt %7 : tensor<1x1x1x64xf32>
    %9 = stablehlo.reshape %cst_2 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<1x1x1x64xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<1x112x112x64xf32>
    %13 = stablehlo.reshape %cst_3 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
    %15 = stablehlo.add %12, %14 : tensor<1x112x112x64xf32>
    %16 = call @relu(%15) : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %cst_100 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_100, dims = [] : (tensor<f32>) -> tensor<f32>
    %18 = "stablehlo.reduce_window"(%16, %17) <{padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %335 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %335 : tensor<f32>
    }) : (tensor<1x112x112x64xf32>, tensor<f32>) -> tensor<1x56x56x64xf32>
    %19 = stablehlo.convolution(%18, %cst_4) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %20 = stablehlo.broadcast_in_dim %cst_5, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %21 = stablehlo.broadcast_in_dim %cst_6, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %22 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %23 = stablehlo.subtract %19, %22 : tensor<1x56x56x64xf32>
    %cst_101 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_101, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %25 = stablehlo.add %21, %24 : tensor<1x1x1x64xf32>
    %26 = stablehlo.rsqrt %25 : tensor<1x1x1x64xf32>
    %27 = stablehlo.reshape %cst_7 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %28 = stablehlo.multiply %26, %27 : tensor<1x1x1x64xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %30 = stablehlo.multiply %23, %29 : tensor<1x56x56x64xf32>
    %31 = stablehlo.reshape %cst_8 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %33 = stablehlo.add %30, %32 : tensor<1x56x56x64xf32>
    %34 = call @relu_0(%33) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %35 = stablehlo.convolution(%34, %cst_9) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %37 = stablehlo.broadcast_in_dim %cst_11, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %38 = stablehlo.broadcast_in_dim %36, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %39 = stablehlo.subtract %35, %38 : tensor<1x56x56x64xf32>
    %cst_102 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %cst_102, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %41 = stablehlo.add %37, %40 : tensor<1x1x1x64xf32>
    %42 = stablehlo.rsqrt %41 : tensor<1x1x1x64xf32>
    %43 = stablehlo.reshape %cst_12 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %44 = stablehlo.multiply %42, %43 : tensor<1x1x1x64xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %46 = stablehlo.multiply %39, %45 : tensor<1x56x56x64xf32>
    %47 = stablehlo.reshape %cst_13 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %48 = stablehlo.broadcast_in_dim %47, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %49 = stablehlo.add %46, %48 : tensor<1x56x56x64xf32>
    %50 = stablehlo.add %49, %18 : tensor<1x56x56x64xf32>
    %51 = call @relu_0(%50) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %52 = stablehlo.convolution(%51, %cst_14) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %54 = stablehlo.broadcast_in_dim %cst_16, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %55 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %56 = stablehlo.subtract %52, %55 : tensor<1x56x56x64xf32>
    %cst_103 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %57 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %58 = stablehlo.add %54, %57 : tensor<1x1x1x64xf32>
    %59 = stablehlo.rsqrt %58 : tensor<1x1x1x64xf32>
    %60 = stablehlo.reshape %cst_17 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %61 = stablehlo.multiply %59, %60 : tensor<1x1x1x64xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %63 = stablehlo.multiply %56, %62 : tensor<1x56x56x64xf32>
    %64 = stablehlo.reshape %cst_18 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %66 = stablehlo.add %63, %65 : tensor<1x56x56x64xf32>
    %67 = call @relu_0(%66) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %68 = stablehlo.convolution(%67, %cst_19) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %69 = stablehlo.broadcast_in_dim %cst_20, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %71 = stablehlo.broadcast_in_dim %69, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %72 = stablehlo.subtract %68, %71 : tensor<1x56x56x64xf32>
    %cst_104 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_104, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %74 = stablehlo.add %70, %73 : tensor<1x1x1x64xf32>
    %75 = stablehlo.rsqrt %74 : tensor<1x1x1x64xf32>
    %76 = stablehlo.reshape %cst_22 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %77 = stablehlo.multiply %75, %76 : tensor<1x1x1x64xf32>
    %78 = stablehlo.broadcast_in_dim %77, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %79 = stablehlo.multiply %72, %78 : tensor<1x56x56x64xf32>
    %80 = stablehlo.reshape %cst_23 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %82 = stablehlo.add %79, %81 : tensor<1x56x56x64xf32>
    %83 = stablehlo.add %82, %51 : tensor<1x56x56x64xf32>
    %84 = call @relu_0(%83) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %85 = stablehlo.convolution(%84, %cst_24) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x64xf32>, tensor<3x3x64x128xf32>) -> tensor<1x28x28x128xf32>
    %86 = stablehlo.broadcast_in_dim %cst_25, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %87 = stablehlo.broadcast_in_dim %cst_26, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %88 = stablehlo.broadcast_in_dim %86, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %89 = stablehlo.subtract %85, %88 : tensor<1x28x28x128xf32>
    %cst_105 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_105, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %91 = stablehlo.add %87, %90 : tensor<1x1x1x128xf32>
    %92 = stablehlo.rsqrt %91 : tensor<1x1x1x128xf32>
    %93 = stablehlo.reshape %cst_27 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %94 = stablehlo.multiply %92, %93 : tensor<1x1x1x128xf32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %96 = stablehlo.multiply %89, %95 : tensor<1x28x28x128xf32>
    %97 = stablehlo.reshape %cst_28 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %98 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %99 = stablehlo.add %96, %98 : tensor<1x28x28x128xf32>
    %100 = call @relu_1(%99) : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %101 = stablehlo.convolution(%100, %cst_29) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
    %102 = stablehlo.broadcast_in_dim %cst_30, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %103 = stablehlo.broadcast_in_dim %cst_31, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %104 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %105 = stablehlo.subtract %101, %104 : tensor<1x28x28x128xf32>
    %cst_106 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %106 = stablehlo.broadcast_in_dim %cst_106, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %107 = stablehlo.add %103, %106 : tensor<1x1x1x128xf32>
    %108 = stablehlo.rsqrt %107 : tensor<1x1x1x128xf32>
    %109 = stablehlo.reshape %cst_32 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %110 = stablehlo.multiply %108, %109 : tensor<1x1x1x128xf32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %112 = stablehlo.multiply %105, %111 : tensor<1x28x28x128xf32>
    %113 = stablehlo.reshape %cst_33 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %114 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %115 = stablehlo.add %112, %114 : tensor<1x28x28x128xf32>
    %116 = stablehlo.convolution(%84, %cst_34) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x64xf32>, tensor<1x1x64x128xf32>) -> tensor<1x28x28x128xf32>
    %117 = stablehlo.broadcast_in_dim %cst_35, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %118 = stablehlo.broadcast_in_dim %cst_36, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %119 = stablehlo.broadcast_in_dim %117, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %120 = stablehlo.subtract %116, %119 : tensor<1x28x28x128xf32>
    %cst_107 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %121 = stablehlo.broadcast_in_dim %cst_107, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %122 = stablehlo.add %118, %121 : tensor<1x1x1x128xf32>
    %123 = stablehlo.rsqrt %122 : tensor<1x1x1x128xf32>
    %124 = stablehlo.reshape %cst_37 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %125 = stablehlo.multiply %123, %124 : tensor<1x1x1x128xf32>
    %126 = stablehlo.broadcast_in_dim %125, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %127 = stablehlo.multiply %120, %126 : tensor<1x28x28x128xf32>
    %128 = stablehlo.reshape %cst_38 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %129 = stablehlo.broadcast_in_dim %128, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %130 = stablehlo.add %127, %129 : tensor<1x28x28x128xf32>
    %131 = stablehlo.add %115, %130 : tensor<1x28x28x128xf32>
    %132 = call @relu_1(%131) : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %133 = stablehlo.convolution(%132, %cst_39) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
    %134 = stablehlo.broadcast_in_dim %cst_40, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %135 = stablehlo.broadcast_in_dim %cst_41, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %136 = stablehlo.broadcast_in_dim %134, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %137 = stablehlo.subtract %133, %136 : tensor<1x28x28x128xf32>
    %cst_108 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %138 = stablehlo.broadcast_in_dim %cst_108, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %139 = stablehlo.add %135, %138 : tensor<1x1x1x128xf32>
    %140 = stablehlo.rsqrt %139 : tensor<1x1x1x128xf32>
    %141 = stablehlo.reshape %cst_42 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %142 = stablehlo.multiply %140, %141 : tensor<1x1x1x128xf32>
    %143 = stablehlo.broadcast_in_dim %142, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %144 = stablehlo.multiply %137, %143 : tensor<1x28x28x128xf32>
    %145 = stablehlo.reshape %cst_43 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %146 = stablehlo.broadcast_in_dim %145, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %147 = stablehlo.add %144, %146 : tensor<1x28x28x128xf32>
    %148 = call @relu_1(%147) : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %149 = stablehlo.convolution(%148, %cst_44) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
    %150 = stablehlo.broadcast_in_dim %cst_45, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %151 = stablehlo.broadcast_in_dim %cst_46, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %152 = stablehlo.broadcast_in_dim %150, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %153 = stablehlo.subtract %149, %152 : tensor<1x28x28x128xf32>
    %cst_109 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %154 = stablehlo.broadcast_in_dim %cst_109, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %155 = stablehlo.add %151, %154 : tensor<1x1x1x128xf32>
    %156 = stablehlo.rsqrt %155 : tensor<1x1x1x128xf32>
    %157 = stablehlo.reshape %cst_47 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %158 = stablehlo.multiply %156, %157 : tensor<1x1x1x128xf32>
    %159 = stablehlo.broadcast_in_dim %158, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %160 = stablehlo.multiply %153, %159 : tensor<1x28x28x128xf32>
    %161 = stablehlo.reshape %cst_48 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %162 = stablehlo.broadcast_in_dim %161, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %163 = stablehlo.add %160, %162 : tensor<1x28x28x128xf32>
    %164 = stablehlo.add %163, %132 : tensor<1x28x28x128xf32>
    %165 = call @relu_1(%164) : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %166 = stablehlo.convolution(%165, %cst_49) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x128xf32>, tensor<3x3x128x256xf32>) -> tensor<1x14x14x256xf32>
    %167 = stablehlo.broadcast_in_dim %cst_50, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %168 = stablehlo.broadcast_in_dim %cst_51, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %169 = stablehlo.broadcast_in_dim %167, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %170 = stablehlo.subtract %166, %169 : tensor<1x14x14x256xf32>
    %cst_110 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %171 = stablehlo.broadcast_in_dim %cst_110, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %172 = stablehlo.add %168, %171 : tensor<1x1x1x256xf32>
    %173 = stablehlo.rsqrt %172 : tensor<1x1x1x256xf32>
    %174 = stablehlo.reshape %cst_52 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %175 = stablehlo.multiply %173, %174 : tensor<1x1x1x256xf32>
    %176 = stablehlo.broadcast_in_dim %175, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %177 = stablehlo.multiply %170, %176 : tensor<1x14x14x256xf32>
    %178 = stablehlo.reshape %cst_53 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %179 = stablehlo.broadcast_in_dim %178, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %180 = stablehlo.add %177, %179 : tensor<1x14x14x256xf32>
    %181 = call @relu_2(%180) : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %182 = stablehlo.convolution(%181, %cst_54) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
    %183 = stablehlo.broadcast_in_dim %cst_55, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %184 = stablehlo.broadcast_in_dim %cst_56, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %185 = stablehlo.broadcast_in_dim %183, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %186 = stablehlo.subtract %182, %185 : tensor<1x14x14x256xf32>
    %cst_111 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %187 = stablehlo.broadcast_in_dim %cst_111, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %188 = stablehlo.add %184, %187 : tensor<1x1x1x256xf32>
    %189 = stablehlo.rsqrt %188 : tensor<1x1x1x256xf32>
    %190 = stablehlo.reshape %cst_57 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %191 = stablehlo.multiply %189, %190 : tensor<1x1x1x256xf32>
    %192 = stablehlo.broadcast_in_dim %191, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %193 = stablehlo.multiply %186, %192 : tensor<1x14x14x256xf32>
    %194 = stablehlo.reshape %cst_58 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %195 = stablehlo.broadcast_in_dim %194, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %196 = stablehlo.add %193, %195 : tensor<1x14x14x256xf32>
    %197 = stablehlo.convolution(%165, %cst_59) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x128xf32>, tensor<1x1x128x256xf32>) -> tensor<1x14x14x256xf32>
    %198 = stablehlo.broadcast_in_dim %cst_60, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %199 = stablehlo.broadcast_in_dim %cst_61, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %200 = stablehlo.broadcast_in_dim %198, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %201 = stablehlo.subtract %197, %200 : tensor<1x14x14x256xf32>
    %cst_112 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %202 = stablehlo.broadcast_in_dim %cst_112, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %203 = stablehlo.add %199, %202 : tensor<1x1x1x256xf32>
    %204 = stablehlo.rsqrt %203 : tensor<1x1x1x256xf32>
    %205 = stablehlo.reshape %cst_62 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %206 = stablehlo.multiply %204, %205 : tensor<1x1x1x256xf32>
    %207 = stablehlo.broadcast_in_dim %206, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %208 = stablehlo.multiply %201, %207 : tensor<1x14x14x256xf32>
    %209 = stablehlo.reshape %cst_63 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %210 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %211 = stablehlo.add %208, %210 : tensor<1x14x14x256xf32>
    %212 = stablehlo.add %196, %211 : tensor<1x14x14x256xf32>
    %213 = call @relu_2(%212) : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %214 = stablehlo.convolution(%213, %cst_64) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
    %215 = stablehlo.broadcast_in_dim %cst_65, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %216 = stablehlo.broadcast_in_dim %cst_66, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %217 = stablehlo.broadcast_in_dim %215, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %218 = stablehlo.subtract %214, %217 : tensor<1x14x14x256xf32>
    %cst_113 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %219 = stablehlo.broadcast_in_dim %cst_113, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %220 = stablehlo.add %216, %219 : tensor<1x1x1x256xf32>
    %221 = stablehlo.rsqrt %220 : tensor<1x1x1x256xf32>
    %222 = stablehlo.reshape %cst_67 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %223 = stablehlo.multiply %221, %222 : tensor<1x1x1x256xf32>
    %224 = stablehlo.broadcast_in_dim %223, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %225 = stablehlo.multiply %218, %224 : tensor<1x14x14x256xf32>
    %226 = stablehlo.reshape %cst_68 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %227 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %228 = stablehlo.add %225, %227 : tensor<1x14x14x256xf32>
    %229 = call @relu_2(%228) : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %230 = stablehlo.convolution(%229, %cst_69) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
    %231 = stablehlo.broadcast_in_dim %cst_70, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %232 = stablehlo.broadcast_in_dim %cst_71, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %233 = stablehlo.broadcast_in_dim %231, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %234 = stablehlo.subtract %230, %233 : tensor<1x14x14x256xf32>
    %cst_114 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %235 = stablehlo.broadcast_in_dim %cst_114, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %236 = stablehlo.add %232, %235 : tensor<1x1x1x256xf32>
    %237 = stablehlo.rsqrt %236 : tensor<1x1x1x256xf32>
    %238 = stablehlo.reshape %cst_72 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %239 = stablehlo.multiply %237, %238 : tensor<1x1x1x256xf32>
    %240 = stablehlo.broadcast_in_dim %239, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %241 = stablehlo.multiply %234, %240 : tensor<1x14x14x256xf32>
    %242 = stablehlo.reshape %cst_73 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %243 = stablehlo.broadcast_in_dim %242, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %244 = stablehlo.add %241, %243 : tensor<1x14x14x256xf32>
    %245 = stablehlo.add %244, %213 : tensor<1x14x14x256xf32>
    %246 = call @relu_2(%245) : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %247 = stablehlo.convolution(%246, %cst_74) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<3x3x256x512xf32>) -> tensor<1x7x7x512xf32>
    %248 = stablehlo.broadcast_in_dim %cst_75, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %249 = stablehlo.broadcast_in_dim %cst_76, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %250 = stablehlo.broadcast_in_dim %248, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %251 = stablehlo.subtract %247, %250 : tensor<1x7x7x512xf32>
    %cst_115 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %252 = stablehlo.broadcast_in_dim %cst_115, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %253 = stablehlo.add %249, %252 : tensor<1x1x1x512xf32>
    %254 = stablehlo.rsqrt %253 : tensor<1x1x1x512xf32>
    %255 = stablehlo.reshape %cst_77 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %256 = stablehlo.multiply %254, %255 : tensor<1x1x1x512xf32>
    %257 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %258 = stablehlo.multiply %251, %257 : tensor<1x7x7x512xf32>
    %259 = stablehlo.reshape %cst_78 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %260 = stablehlo.broadcast_in_dim %259, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %261 = stablehlo.add %258, %260 : tensor<1x7x7x512xf32>
    %262 = call @relu_3(%261) : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %263 = stablehlo.convolution(%262, %cst_79) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
    %264 = stablehlo.broadcast_in_dim %cst_80, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %265 = stablehlo.broadcast_in_dim %cst_81, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %266 = stablehlo.broadcast_in_dim %264, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %267 = stablehlo.subtract %263, %266 : tensor<1x7x7x512xf32>
    %cst_116 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %268 = stablehlo.broadcast_in_dim %cst_116, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %269 = stablehlo.add %265, %268 : tensor<1x1x1x512xf32>
    %270 = stablehlo.rsqrt %269 : tensor<1x1x1x512xf32>
    %271 = stablehlo.reshape %cst_82 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %272 = stablehlo.multiply %270, %271 : tensor<1x1x1x512xf32>
    %273 = stablehlo.broadcast_in_dim %272, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %274 = stablehlo.multiply %267, %273 : tensor<1x7x7x512xf32>
    %275 = stablehlo.reshape %cst_83 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %276 = stablehlo.broadcast_in_dim %275, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %277 = stablehlo.add %274, %276 : tensor<1x7x7x512xf32>
    %278 = stablehlo.convolution(%246, %cst_84) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<1x1x256x512xf32>) -> tensor<1x7x7x512xf32>
    %279 = stablehlo.broadcast_in_dim %cst_85, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %280 = stablehlo.broadcast_in_dim %cst_86, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %281 = stablehlo.broadcast_in_dim %279, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %282 = stablehlo.subtract %278, %281 : tensor<1x7x7x512xf32>
    %cst_117 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %283 = stablehlo.broadcast_in_dim %cst_117, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %284 = stablehlo.add %280, %283 : tensor<1x1x1x512xf32>
    %285 = stablehlo.rsqrt %284 : tensor<1x1x1x512xf32>
    %286 = stablehlo.reshape %cst_87 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %287 = stablehlo.multiply %285, %286 : tensor<1x1x1x512xf32>
    %288 = stablehlo.broadcast_in_dim %287, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %289 = stablehlo.multiply %282, %288 : tensor<1x7x7x512xf32>
    %290 = stablehlo.reshape %cst_88 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %291 = stablehlo.broadcast_in_dim %290, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %292 = stablehlo.add %289, %291 : tensor<1x7x7x512xf32>
    %293 = stablehlo.add %277, %292 : tensor<1x7x7x512xf32>
    %294 = call @relu_3(%293) : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %295 = stablehlo.convolution(%294, %cst_89) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
    %296 = stablehlo.broadcast_in_dim %cst_90, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %297 = stablehlo.broadcast_in_dim %cst_91, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %298 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %299 = stablehlo.subtract %295, %298 : tensor<1x7x7x512xf32>
    %cst_118 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %300 = stablehlo.broadcast_in_dim %cst_118, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %301 = stablehlo.add %297, %300 : tensor<1x1x1x512xf32>
    %302 = stablehlo.rsqrt %301 : tensor<1x1x1x512xf32>
    %303 = stablehlo.reshape %cst_92 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %304 = stablehlo.multiply %302, %303 : tensor<1x1x1x512xf32>
    %305 = stablehlo.broadcast_in_dim %304, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %306 = stablehlo.multiply %299, %305 : tensor<1x7x7x512xf32>
    %307 = stablehlo.reshape %cst_93 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %309 = stablehlo.add %306, %308 : tensor<1x7x7x512xf32>
    %310 = call @relu_3(%309) : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %311 = stablehlo.convolution(%310, %cst_94) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
    %312 = stablehlo.broadcast_in_dim %cst_95, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %313 = stablehlo.broadcast_in_dim %cst_96, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %314 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %315 = stablehlo.subtract %311, %314 : tensor<1x7x7x512xf32>
    %cst_119 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %316 = stablehlo.broadcast_in_dim %cst_119, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %317 = stablehlo.add %313, %316 : tensor<1x1x1x512xf32>
    %318 = stablehlo.rsqrt %317 : tensor<1x1x1x512xf32>
    %319 = stablehlo.reshape %cst_97 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %320 = stablehlo.multiply %318, %319 : tensor<1x1x1x512xf32>
    %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %322 = stablehlo.multiply %315, %321 : tensor<1x7x7x512xf32>
    %323 = stablehlo.reshape %cst_98 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %325 = stablehlo.add %322, %324 : tensor<1x7x7x512xf32>
    %326 = stablehlo.add %325, %294 : tensor<1x7x7x512xf32>
    %327 = call @relu_3(%326) : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %cst_120 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %328 = stablehlo.broadcast_in_dim %cst_120, dims = [] : (tensor<f32>) -> tensor<f32>
    %329 = "stablehlo.reduce_window"(%327, %328) <{window_dimensions = array<i64: 1, 7, 7, 1>, window_strides = array<i64: 1, 7, 7, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %335 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %335 : tensor<f32>
    }) : (tensor<1x7x7x512xf32>, tensor<f32>) -> tensor<1x1x1x512xf32>
    %c = stablehlo.constant dense<49> : tensor<i32>
    %330 = stablehlo.convert %c : (tensor<i32>) -> tensor<f32>
    %331 = stablehlo.broadcast_in_dim %330, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %332 = stablehlo.divide %329, %331 : tensor<1x1x1x512xf32>
    %333 = stablehlo.transpose %332, dims = [0, 3, 1, 2] : (tensor<1x1x1x512xf32>) -> tensor<1x512x1x1xf32>
    %334 = stablehlo.transpose %327, dims = [0, 3, 1, 2] : (tensor<1x7x7x512xf32>) -> tensor<1x512x7x7xf32>
    return %334, %333 : tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>
  }
  func.func private @relu(%arg0: tensor<1x112x112x64xf32> {mhlo.layout_mode = "default"}) -> (tensor<1x112x112x64xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x112x112x64xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x112x112x64xf32>
    return %1 : tensor<1x112x112x64xf32>
  }
  func.func private @relu_0(%arg0: tensor<1x56x56x64xf32> {mhlo.layout_mode = "default"}) -> (tensor<1x56x56x64xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56x64xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x56x56x64xf32>
    return %1 : tensor<1x56x56x64xf32>
  }
  func.func private @relu_1(%arg0: tensor<1x28x28x128xf32> {mhlo.layout_mode = "default"}) -> (tensor<1x28x28x128xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x28x28x128xf32>
    return %1 : tensor<1x28x28x128xf32>
  }
  func.func private @relu_2(%arg0: tensor<1x14x14x256xf32> {mhlo.layout_mode = "default"}) -> (tensor<1x14x14x256xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14x256xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x14x14x256xf32>
    return %1 : tensor<1x14x14x256xf32>
  }
  func.func private @relu_3(%arg0: tensor<1x7x7x512xf32> {mhlo.layout_mode = "default"}) -> (tensor<1x7x7x512xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7x512xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x7x7x512xf32>
    return %1 : tensor<1x7x7x512xf32>
  }
}