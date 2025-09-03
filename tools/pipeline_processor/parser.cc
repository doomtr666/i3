#include "parser.h"

static const char* grammar_ = R"x(
# full pipeline definition
file <- file_stmt*

# pipeline stmt
file_stmt <- pipeline

# pipeline
pipeline <- pipeline_kw id l_brace pipeline_stmt* r_brace

# pipeline stmt
pipeline_stmt <- compile_stmt

compile_stmt <- compile_kw l_paren id comma id r_paren semicolon

# keywords
pipeline_kw <- "pipeline"
compile_kw <- "compile"

# identifier
id <- <[a-zA-Z_][a-zA-Z0-9_]*>

# punct
comma <- ","
semicolon <- ";"
l_paren <- "("
r_paren <- ")"
l_brace <- "{"
r_brace <- "}"

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