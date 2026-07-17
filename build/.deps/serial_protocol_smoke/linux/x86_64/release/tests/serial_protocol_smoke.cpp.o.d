{
    files = {
        "tests/serial_protocol_smoke.cpp"
    },
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
    depfiles = "serial_protocol_smoke.o: tests/serial_protocol_smoke.cpp  include/l1_sensor/serial/serial_protocol.hpp  include/l1_sensor/serial/robot_state.hpp  include/l5_control/serial_command.hpp\
",
    depfiles_format = "gcc"
}