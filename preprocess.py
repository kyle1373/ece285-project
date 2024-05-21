import os
import struct
from PIL import Image

def is_chinese_character(char):
    """Check if a character is a Chinese character."""
    return (
        '\u4e00' <= char <= '\u9fff' or
        '\u3400' <= char <= '\u4dbf' or
        '\u20000' <= char <= '\u2a6df'
    )

def read_gnt_file(file_path):
    with open(file_path, 'rb') as f:
        file_length = os.path.getsize(file_path)
        while f.tell() < file_length:
            char_length = struct.unpack('<I', f.read(4))[0]
            char_label = f.read(2).decode('gbk', errors='ignore')
            char_width = struct.unpack('<H', f.read(2))[0]
            char_height = struct.unpack('<H', f.read(2))[0]
            char_image = f.read(char_width * char_height)

            if is_chinese_character(char_label):
                image = Image.frombytes('L', (char_width, char_height), char_image)
                yield char_label, image

def sanitize_filename(tag_code, i):
    sanitized_tag_code = ''.join(c if c.isalnum() else '_' for c in tag_code)
    return f"{sanitized_tag_code}_{i}.png"

def extract_images(gnt_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            for i, (tag_code, image) in enumerate(read_gnt_file(file_path)):
                image_file_name = sanitize_filename(tag_code, i)
                image_path = os.path.join(output_dir, image_file_name)
                image.save(image_path)