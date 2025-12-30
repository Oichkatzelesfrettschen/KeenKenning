import re

file_path = 'app/src/main/jni/keen.c'
with open(file_path, 'r') as f:
    content = f.read()

# List of functions to mark unused
funcs = [
    'get_maxblk', 'default_params', 'game_fetch_preset', 'free_params',
    'dup_params', 'decode_params', 'encode_params', 'game_configure',
    'custom_params', 'validate_params', 'encode_block_structure',
    'parse_block_structure', 'recalculate_clues'
]

for func in funcs:
    # Regex to match "static [inline] [type] func("
    # We capture everything from 'static' up to the function name
    # and insert the attribute before the function name.
    
    # Matches: static inline int get_maxblk
    # Matches: static game_params* default_params
    # Matches: static char* encode_block_structure
    
    pattern = r'(static\s+(?:inline\s+)?(?:[\w]+\s*\*?\s*)+)\b' + func + r'\s*\('
    replacement = r'\1__attribute__((unused)) ' + func + r'('
    
    content = re.sub(pattern, replacement, content)

with open(file_path, 'w') as f:
    f.write(content)
