{
    files = {
        "../sp_vision_25-main/io/serial/src/serial.cc"
    },
    depfiles_format = "gcc",
    depfiles = "serial.o: ../sp_vision_25-main/io/serial/src/serial.cc  ../sp_vision_25-main/io/serial/include/serial/serial.h  ../sp_vision_25-main/io/serial/include/serial/v8stdint.h  ../sp_vision_25-main/io/serial/include/serial/impl/unix.h\
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
    }
}