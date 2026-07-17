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
            "-Itools/logger/include",
            "-Itools/logger/include/3rdparty",
            "-DNDEBUG"
        }
    },
    depfiles = "serial_protocol.o: src/l1_sensor/serial/serial_protocol.cpp  include/l1_sensor/serial/serial_protocol.hpp  include/l1_sensor/serial/robot_state.hpp  include/l5_control/serial_command.hpp include/l6_telemetry/logger.hpp\
",
    files = {
        "src/l1_sensor/serial/serial_protocol.cpp"
    },
    depfiles_format = "gcc"
}