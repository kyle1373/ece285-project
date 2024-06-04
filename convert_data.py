import os
import sys
import cairosvg
from bs4 import BeautifulSoup

def remove_keyframes(svg_content):
    soup = BeautifulSoup(svg_content, 'xml')

    # Remove <style> elements containing keyframes
    for style in soup.find_all('style'):
        style.decompose()

    # Remove animation attributes from elements
    for elem in soup.find_all(['path', 'g']):
        if 'animation' in elem.attrs:
            del elem.attrs['animation']
        if 'animation-timing-function' in elem.attrs:
            del elem.attrs['animation-timing-function']
        if 'animation-delay' in elem.attrs:
            del elem.attrs['animation-delay']

    return str(soup)

def convert_svg_to_png(svg_path, png_path):
    with open(svg_path, 'r', encoding='utf-8') as svg_file:
        svg_content = svg_file.read()
    
    final_svg_content = remove_keyframes(svg_content)
    
    cairosvg.svg2png(bytestring=final_svg_content.encode('utf-8'), write_to=png_path)

def convert_svgs_in_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    length = len(os.listdir(input_dir))
    curr_idx = 0
    
    for file_name in os.listdir(input_dir):
        curr_idx = curr_idx + 1
        if file_name.endswith('.svg'):
            svg_path = os.path.join(input_dir, file_name)
            png_path = os.path.join(output_dir, file_name.replace('.svg', '.png'))
            convert_svg_to_png(svg_path, png_path)
            print(f'({curr_idx}/{length}) Converted {file_name} to PNG')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    convert_svgs_in_directory(input_dir, output_dir)