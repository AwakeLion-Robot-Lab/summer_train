{
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
            "-I../sp_vision_25-main/io/serial/include",
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
            "-DNDEBUG"
        }
    },
    files = {
        "src/l1_sensor/serial_protocol.cpp"
    },
    depfiles = "serial_protocol.o: src/l1_sensor/serial_protocol.cpp  include/l1_sensor/serial_protocol.hpp include/l1_sensor/robot_state.hpp  include/l5_control/serial_command.hpp\
"
}