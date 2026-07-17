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
    depfiles = "serial_port.o: src/l1_sensor/serial_port.cpp  include/l1_sensor/serial_port.hpp include/l1_sensor/serial_config.hpp  ../sp_vision_25-main/io/serial/include/serial/serial.h  ../sp_vision_25-main/io/serial/include/serial/v8stdint.h\
",
    files = {
        "src/l1_sensor/serial_port.cpp"
    },
    depfiles_format = "gcc"
}