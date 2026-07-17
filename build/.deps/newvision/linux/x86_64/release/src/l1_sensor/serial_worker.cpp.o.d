{
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
    depfiles = "serial_worker.o: src/l1_sensor/serial_worker.cpp  include/l1_sensor/serial_worker.hpp include/l1_sensor/robot_state.hpp  include/l1_sensor/serial_config.hpp include/l1_sensor/serial_port.hpp  include/l1_sensor/serial_protocol.hpp  include/l5_control/serial_command.hpp\
",
    files = {
        "src/l1_sensor/serial_worker.cpp"
    },
    depfiles_format = "gcc"
}