{
    depfiles = "robot_state.o: src/l1_sensor/serial/robot_state.cpp  include/l1_sensor/serial/robot_state.hpp\
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
        "src/l1_sensor/serial/robot_state.cpp"
    }
}