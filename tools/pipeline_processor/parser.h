#pragma once

#include "peglib.h"

class parser
{
    peg::parser parser_;

  public:
    parser();

    std::shared_ptr<peg::Ast> parse(const char* source);
};