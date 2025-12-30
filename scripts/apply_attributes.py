import re

file_path = 'app/src/main/jni/keen.c'
with open(file_path, 'r') as f:
    content = f.read()

funcs = [
    'get_maxblk', 'default_params', 'game_fetch_preset', 'free_params',
    'dup_params', 'decode_params', 'encode_params', 'game_configure',
    'custom_params', 'validate_params', 'encode_block_structure',
    'parse_block_structure', 'recalculate_clues'
]

# We want to prepend [[maybe_unused]] to the static declaration.
# We look for "static ... func("
# And replace "static" with "[[maybe_unused]] static"

# But we only want to do this for the specific functions.
# Regex: r'(static\s+.*?\b' + func + r'\s*\()'
# Replacement: r'[[maybe_unused]] \1'

# We also need to clean up any __attribute__((unused)) we might have half-added or missed.
content = content.replace('__attribute__((unused))', '')

count = 0
for func in funcs:
    # Match "static" at start of line (possibly indented) followed by return type, then func name
    # We use non-greedy matching .*? until the function name
    pattern = r'(\bstatic\b.*?\b' + func + r'\s*\()'
    
    # Check if already has [[maybe_unused]] (to be safe, though we shouldn't need to if we start clean)
    if f'[[maybe_unused]] static' in content:
         # Skip if already done? No, might be different function.
         pass

    # Perform replacement
    # We assume 'static' is the first word of the declaration
    # We replace 'static' with '[[maybe_unused]] static' within that specific match
    
    # Actually, re.sub with a function is safer.
    def replace_match(match):
        s = match.group(1)
        if '[[maybe_unused]]' in s:
            return s
        return '[[maybe_unused]] ' + s

    new_content, n = re.subn(pattern, replace_match, content)
    if n > 0:
        print(f"Patched {func}")
        content = new_content
        count += n
    else:
        print(f"Could not find {func}")

print(f"Total patches: {count}")

with open(file_path, 'w') as f:
    f.write(content)
