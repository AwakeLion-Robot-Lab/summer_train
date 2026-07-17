{
    depfiles_format = "gcc",
    depfiles = "logger.o: src/l6_telemetry/logger.cpp include/l6_telemetry/logger.hpp  tools/logger/include/aw_logger/aw_logger.hpp  tools/logger/include/aw_logger/appender.hpp  tools/logger/include/aw_logger/exception.hpp  tools/logger/include/aw_logger/formatter.hpp  tools/logger/include/3rdparty/nlohmann/json.hpp  tools/logger/include/aw_logger/log_event.hpp  tools/logger/include/aw_logger/fmt_base.hpp  tools/logger/include/aw_logger/logger.hpp  tools/logger/include/aw_logger/ring_buffer.hpp  tools/logger/include/aw_logger/log_macro.hpp  tools/logger/include/aw_logger/impl/console_appender_impl.hpp  tools/logger/include/aw_logger/impl/file_appender_impl.hpp  tools/logger/include/aw_logger/impl/formatter_impl.hpp  tools/logger/include/aw_logger/impl/log_event_impl.hpp  tools/logger/include/aw_logger/impl/logger_impl.hpp  tools/logger/include/aw_logger/impl/ring_buffer_impl.hpp\
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
            "-Itools/logger/include",
            "-Itools/logger/include/3rdparty",
            "-DNDEBUG"
        }
    },
    files = {
        "src/l6_telemetry/logger.cpp"
    }
}