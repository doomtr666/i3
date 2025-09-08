#pragma once

#include "peglib.h"

class parser
{
    std::string path_;
    peg::parser parser_;

    void error(const std::string& path, size_t line, size_t col, const std::string& message);

  public:
    parser();
    std::shared_ptr<peg::Ast> parse(const char* source, const char* path = nullptr);
};