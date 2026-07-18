{
    depfiles = "trace_scope.o: src/l6_telemetry/trace_scope.cpp  include/l6_telemetry/trace_scope.hpp include/l6_telemetry/logger.hpp\
",
    files = {
        "src/l6_telemetry/trace_scope.cpp"
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
            "-I/usr/lib/x86_64-linux-gnu/pkgconfig/../../../include/ie",
            "-I/usr/lib/x86_64-linux-gnu/pkgconfig/../../../include",
            "-DTBB_PREVIEW_WAITING_FOR_WORKERS=1",
            "-DIE_THREAD=IE_THREAD_TBB",
            "-DOV_THREAD=OV_THREAD_TBB",
            "-DNEWVISION_HAS_OPENVINO=1",
            "-DNDEBUG"
        }
    }
}