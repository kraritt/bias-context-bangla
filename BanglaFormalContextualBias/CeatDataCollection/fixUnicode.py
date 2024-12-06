import re

def replace_matching_pattern(input_file, output_file, pattern, replacement):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    modified_lines = []

    for line in lines:
        modified_line = re.sub(pattern, replacement, line)
        modified_lines.append(modified_line)

    modified_text = ''.join(modified_lines)

    with open(output_file, 'w') as file:
        file.write(modified_text)

file_path = './banglaWords/ner_static.txt'
output_file_path = './banglaWords/ner_static_mod.txt'
pattern = r"য়"
replacement = 'য়'

replace_matching_pattern(file_path, output_file_path, pattern, replacement)




