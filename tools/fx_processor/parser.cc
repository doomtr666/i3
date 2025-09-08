#include "parser.h"

static const char* grammar_ = R"x(
# file definition
file <- file_stmt* { no_ast_opt }

# file stmt
file_stmt <-
    import
    / parameter
    / slang
    / pipeline 

# import
import <- "import" string_literal ";" { no_ast_opt }

# parameter
parameter <- annotations parameter_type id ";"

# slang section
slang <- "#slang" <((!"#end") .)*> "#end"

# pipeline
pipeline <- ( "graphics" / "compute" ) id "{" pipeline_stmt* "}"

# pipeline stmt
pipeline_stmt <- pipeline_var_stmt
pipeline_var_stmt <- id "=" pipeline_value
pipeline_value <- float_literal / int_literal / bool_literal / string_literal / id / pipeline_array / pipeline_dict
pipeline_array <- "[" COMMA_LIST(pipeline_value, ",")? "]"
pipeline_dict <- "{" COMMA_LIST(pipeline_dict_value, ",")?  "}"
pipeline_dict_value <- id "=" pipeline_value

# annotations
annotations <- annotation*
annotation <- "[" <(!("["/"]") .)*> "]"

# macros
COMMA_LIST(E,S) <- (E (S E)*)

# parameter type
parameter_type <- "float2" / "float3" / "float4"

# literals
int_literal <- <[0-9]+>
bool_literal <- "true" / "false"
float_literal <- <[0-9]+ "." [0-9]*> "f"?
string_literal <- "\"" <(!"\"" .)*> "\""

# identifier
id <- <[a-zA-Z_][a-zA-Z0-9_]*>

#white spaces
comment <- "//" [^\n]* / "/*" (!"*/" .)* "*/"
_ <-  ([ \t\r\n] / comment)*
%whitespace <- _ 
)x";

parser::parser()
{
    parser_.set_logger([&](size_t line, size_t col, const std::string& message) { error(path_, line, col, message); });
    parser_.load_grammar(grammar_);
    parser_.enable_packrat_parsing();
    parser_.enable_ast();
}

void parser::error(const std::string& path, size_t line, size_t col, const std::string& message)
{
    std::cerr << path << ":" << line << ":" << col << ": " << message << std::endl;
}

void parser::error(const std::shared_ptr<peg::Ast>& node, const std::string& message)
{
    error(path_, node->line, node->column, message);
}

std::shared_ptr<peg::Ast> parser::parse(const char* source, const char* path)
{
    if (path == nullptr)
        path_ = "input";
    else
        path_ = path;

    std::shared_ptr<peg::Ast> ast;
    if (parser_.parse(source, ast, path))
        return parser_.optimize_ast(ast);

    return nullptr;
}