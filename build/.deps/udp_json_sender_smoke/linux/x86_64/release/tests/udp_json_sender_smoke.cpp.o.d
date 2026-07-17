{
    files = {
        "tests/udp_json_sender_smoke.cpp"
    },
    depfiles = "udp_json_sender_smoke.o: tests/udp_json_sender_smoke.cpp  include/l6_telemetry/udp_json_sender.hpp  tools/logger/include/3rdparty/nlohmann/json.hpp\
",
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
            "-Itools/logger/include/3rdparty",
            "-DNDEBUG"
        }
    }
}