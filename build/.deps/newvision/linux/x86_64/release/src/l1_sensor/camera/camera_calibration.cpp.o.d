{
    depfiles = "camera_calibration.o: src/l1_sensor/camera/camera_calibration.cpp  include/l1_sensor/camera/camera_calibration.hpp  /usr/include/opencv4/opencv2/core.hpp  /usr/include/opencv4/opencv2/core/cvdef.h  /usr/include/opencv4/opencv2/core/version.hpp  /usr/include/opencv4/opencv2/core/hal/interface.h  /usr/include/opencv4/opencv2/core/cv_cpu_dispatch.h  /usr/include/opencv4/opencv2/core/base.hpp  /usr/include/opencv4/opencv2/opencv_modules.hpp  /usr/include/opencv4/opencv2/core/cvstd.hpp  /usr/include/opencv4/opencv2/core/cvstd_wrapper.hpp  /usr/include/opencv4/opencv2/core/neon_utils.hpp  /usr/include/opencv4/opencv2/core/vsx_utils.hpp  /usr/include/opencv4/opencv2/core/check.hpp  /usr/include/opencv4/opencv2/core/traits.hpp  /usr/include/opencv4/opencv2/core/matx.hpp  /usr/include/opencv4/opencv2/core/saturate.hpp  /usr/include/opencv4/opencv2/core/fast_math.hpp  /usr/include/opencv4/opencv2/core/types.hpp  /usr/include/opencv4/opencv2/core/mat.hpp  /usr/include/opencv4/opencv2/core/bufferpool.hpp  /usr/include/opencv4/opencv2/core/mat.inl.hpp  /usr/include/opencv4/opencv2/core/persistence.hpp  /usr/include/opencv4/opencv2/core/operations.hpp  /usr/include/opencv4/opencv2/core/cvstd.inl.hpp  /usr/include/opencv4/opencv2/core/utility.hpp  /usr/include/opencv4/opencv2/core/optim.hpp  /usr/include/opencv4/opencv2/core/ovx.hpp  /usr/include/opencv4/opencv2/core/cvdef.h  include/l6_telemetry/logger.hpp\
",
    files = {
        "src/l1_sensor/camera/camera_calibration.cpp"
    },
    depfiles_format = "gcc",
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
            "-I/home/rm/rm_summmer/tongjiceshi/.deps/l_openvino_toolkit_ubuntu24_2024.6.0.17404.4c0f47d2335_x86_64/runtime/lib/intel64/pkgconfig/../../../../runtime/include",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
            "-DOV_THREAD=OV_THREAD_TBB",
            "-DNEWVISION_HAS_OPENVINO=1",
            "-DNDEBUG"
        }
    }
}