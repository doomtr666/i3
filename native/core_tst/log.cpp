#include "gtest/gtest.h"

extern "C"
{
#include "native/core/log.h"  
}

TEST(log, get_logger)
{
    i3_logger_i* logger1 = i3_get_logger("test");
    EXPECT_NE(logger1, nullptr);

    // loggers are singletons
    i3_logger_i* logger2 = i3_get_logger("test");
    EXPECT_NE(logger2, nullptr);

    EXPECT_EQ(logger1, logger2);
}

TEST(log, level)
{
    // inf
    i3_logger_i* logger1 = i3_get_logger("test");

    i3_log_dbg(logger1, "debug");
    i3_log_inf(logger1, "info");
    i3_log_wrn(logger1, "warn");
    i3_log_err(logger1, "error");

    logger1->set_level(logger1->self, I3_LOG_LEVEL_DEBUG);

    i3_log_dbg(logger1, "debug");
    i3_log_inf(logger1, "info");
    i3_log_wrn(logger1, "warn");
    i3_log_err(logger1, "error");
}