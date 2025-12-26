# Code Formatting Styles

This document defines the formatting standards for each language in KeenKeenForAndroid.

## Java

**Tool**: Android Lint + Google Java Format (implied)
**Config**: `app/build.gradle` with `warningsAsErrors = true`

```java
// Style: Google Java Style
// Indentation: 4 spaces
// Line length: 100 characters
// Braces: same-line opening

public class Example {
    private static final String TAG = "Example";

    public void method() {
        if (condition) {
            // body
        }
    }
}
```

## Kotlin

**Tool**: ktlint
**Config**: `.editorconfig`

```kotlin
// Style: Kotlin Coding Conventions
// Indentation: 4 spaces
// Line length: 120 characters
// Trailing commas: yes

data class Example(
    val field1: String,
    val field2: Int,
)

fun example() {
    listOf(1, 2, 3).forEach { item ->
        println(item)
    }
}
```

## C/C++

**Tool**: clang-format
**Config**: `.clang-format`

```c
// Style: LLVM with modifications
// Indentation: 4 spaces
// Line length: 100 characters
// Braces: same-line for functions

static void example(int param) {
    if (param > 0) {
        /* body */
    }
}
```

## Python

**Tool**: ruff (format + lint)
**Config**: `pyproject.toml`

```python
# Style: PEP 8 / Black
# Indentation: 4 spaces
# Line length: 88 characters
# Quotes: double

def example(param: int) -> str:
    """Docstring with description."""
    if param > 0:
        return "positive"
    return "non-positive"
```

## XML (Android Resources)

**Tool**: Android Studio formatter / xmllint
**Config**: Inline

```xml
<?xml version="1.0" encoding="utf-8"?>
<resources>
    <!-- Indentation: 4 spaces -->
    <!-- Attributes: one per line for complex elements -->
    <string name="app_name">KeenKeen</string>
</resources>
```

## Shell Scripts

**Tool**: shellcheck
**Config**: `.shellcheckrc`

```bash
#!/bin/bash
# Style: POSIX-compatible where possible
# Indentation: 4 spaces
# Quoting: always quote variables

set -eu

main() {
    local var="$1"
    echo "${var}"
}
```

## EditorConfig

All editors should respect `.editorconfig`:

```ini
root = true

[*]
indent_style = space
end_of_line = lf
charset = utf-8
trim_trailing_whitespace = true
insert_final_newline = true

[*.{java,kt,kts}]
indent_size = 4
max_line_length = 120

[*.{c,h}]
indent_size = 4
max_line_length = 100

[*.py]
indent_size = 4
max_line_length = 88

[*.{xml,gradle}]
indent_size = 4

[Makefile]
indent_style = tab
```
