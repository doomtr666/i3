#include "parser.h"

static const char* grammar_ = R"x(
# file definition
file <- file_stmt*

# file stmt
file_stmt <- pipeline / slang

# slang section
slang <- "#slang" <((!"#end") .)*> "#end"

# pipeline
pipeline <- "pipeline" id "{" pipeline_stmt* "}"

# pipeline stmt
pipeline_stmt <- pipeline_compile_stmt

# compile shader stmt
pipeline_compile_stmt <- "compile" "(" id "," id ")" ";"

# keywords

# identifier
id <- <[a-zA-Z_][a-zA-Z0-9_]*>

#white spaces
comment <- "//" [^\n]* / "/*" (!"*/" .)* "*/"
_ <-  ([ \t\r\n] / comment)*
%whitespace <- _ 
)x";

static void parser_log_error(size_t line, size_t col, const std::string& message)
{
    std::cerr << line << ":" << col << ": " << message << std::endl;
}

parser::parser()
{
    parser_.set_logger(parser_log_error);
    parser_.load_grammar(grammar_);
    parser_.enable_packrat_parsing();
    parser_.enable_ast();
}

std::shared_ptr<peg::Ast> parser::parse(const char* source)
{
    std::shared_ptr<peg::Ast> ast;
    if (parser_.parse(source, ast))
        return parser_.optimize_ast(ast);

    return nullptr;
}