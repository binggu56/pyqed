const tokenPattern =
  /((?:[rubfRUBF]{0,2})(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|#[^\n]*|\b(?:and|as|assert|break|class|continue|def|del|elif|else|except|False|finally|for|from|global|if|import|in|is|lambda|None|nonlocal|not|or|pass|raise|return|True|try|while|with|yield)\b|\b(?:enumerate|float|int|len|Path|print|range|RuntimeError|str|zip)\b|\b\d+(?:\.\d+)?\b)/g;

function tokenClass(token: string) {
  if (/^#/.test(token)) return "syntax-comment";
  if (/^[rubfRUBF]{0,2}["']/.test(token)) return "syntax-string";
  if (/^\d/.test(token)) return "syntax-number";
  if (/^(?:enumerate|float|int|len|Path|print|range|RuntimeError|str|zip)$/.test(token)) {
    return "syntax-builtin";
  }
  if (
    /^(?:and|as|assert|break|class|continue|def|del|elif|else|except|False|finally|for|from|global|if|import|in|is|lambda|None|nonlocal|not|or|pass|raise|return|True|try|while|with|yield)$/.test(
      token,
    )
  ) {
    return "syntax-keyword";
  }
  return null;
}

export function PythonCode({ code }: { code: string }) {
  return (
    <code className="python-code">
      {code.split("\n").map((line, lineIndex) => {
        const tokens = line.split(tokenPattern).filter(Boolean);
        return (
          <span className="code-line" key={`${lineIndex}-${line}`}>
            {tokens.length === 0
              ? "\u00a0"
              : tokens.map((token, tokenIndex) => {
                  const className = tokenClass(token);
                  return className ? (
                    <span
                      className={className}
                      key={`${lineIndex}-${tokenIndex}`}
                    >
                      {token}
                    </span>
                  ) : (
                    token
                  );
                })}
          </span>
        );
      })}
    </code>
  );
}
