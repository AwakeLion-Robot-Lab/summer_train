{
    depfiles = "serial_protocol.o: src/l1_sensor/serial/serial_protocol.cpp  include/l1_sensor/serial/serial_protocol.hpp  include/l1_sensor/serial/robot_state.hpp  include/l5_control/serial_command.hpp include/l6_telemetry/logger.hpp\
",
    files = {
        "src/l1_sensor/serial/serial_protocol.cpp"
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