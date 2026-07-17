{
    depfiles = "armor_decoder.o: src/l2_perception/armor/armor_decoder.cpp  include/l2_perception/armor/armor_decoder.hpp  include/l2_perception/armor.hpp  /usr/include/opencv4/opencv2/core/types.hpp  /usr/include/opencv4/opencv2/core/cvdef.h  /usr/include/opencv4/opencv2/core/version.hpp  /usr/include/opencv4/opencv2/core/hal/interface.h  /usr/include/opencv4/opencv2/core/cv_cpu_dispatch.h  /usr/include/opencv4/opencv2/core/cvstd.hpp  /usr/include/opencv4/opencv2/core/cvstd_wrapper.hpp  /usr/include/opencv4/opencv2/core/matx.hpp  /usr/include/opencv4/opencv2/core/base.hpp  /usr/include/opencv4/opencv2/opencv_modules.hpp  /usr/include/opencv4/opencv2/core/neon_utils.hpp  /usr/include/opencv4/opencv2/core/vsx_utils.hpp  /usr/include/opencv4/opencv2/core/check.hpp  /usr/include/opencv4/opencv2/core/traits.hpp  /usr/include/opencv4/opencv2/core/saturate.hpp  /usr/include/opencv4/opencv2/core/fast_math.hpp  include/l2_perception/inference/image_preprocessor.hpp  include/l2_perception/inference/inference_backend.hpp  include/l2_perception/inference/inference_result.hpp  /usr/include/opencv4/opencv2/core.hpp  /usr/include/opencv4/opencv2/core/mat.hpp  /usr/include/opencv4/opencv2/core/bufferpool.hpp  /usr/include/opencv4/opencv2/core/mat.inl.hpp  /usr/include/opencv4/opencv2/core/persistence.hpp  /usr/include/opencv4/opencv2/core/operations.hpp  /usr/include/opencv4/opencv2/core/cvstd.inl.hpp  /usr/include/opencv4/opencv2/core/utility.hpp  /usr/include/opencv4/opencv2/core/optim.hpp  /usr/include/opencv4/opencv2/core/ovx.hpp  /usr/include/opencv4/opencv2/core/cvdef.h\
",
    values = {
        "/usr/bin/g++-13",
        {
            "-m64",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-O3",
            "-std=c++20",
            "-Iinclude",
            "-I/usr/include/eigen3",
            "-Itools/serial/include",
            "-Itools/config_set/include",
            "-Itools/LatesBuffer/include",
            "-Itools/camera_sdk",
            "-Itools/camera_sdk/hikrobot",
            "-Itools/camera_sdk/hikrobot/include",
            "-Itools/camera_sdk/mindvision",
            "-Itools/camera_sdk/mindvision/include",
            "-Itools/logger/include",
            "-Itools/logger/include/3rdparty",
            "-I/usr/include/opencv4",
            "-I/opt/intel/openvino2023/runtime/lib/intel64/pkgconfig/../../../../runtime/include/ie",
            "-I/opt/intel/openvino2023/runtime/lib/intel64/pkgconfig/../../../../runtime/include",
            "-DTBB_PREVIEW_WAITING_FOR_WORKERS=1",
            "-DIE_THREAD=IE_THREAD_TBB",
            "-DOV_THREAD=OV_THREAD_TBB",
            "-DNEWVISION_HAS_OPENVINO=1",
            "-DNDEBUG"
        }
    },
    depfiles_format = "gcc",
    files = {
        "src/l2_perception/armor/armor_decoder.cpp"
    }
}