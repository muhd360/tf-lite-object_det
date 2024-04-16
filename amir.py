import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size').find('width').text),
                     int(root.find('size').find('height').text),
                     member.find('name').text,
                     int(member.find('bndbox').find('xmin').text),
                     int(member.find('bndbox').find('ymin').text),
                     int(member.find('bndbox').find('xmax').text),
                     int(member.find('bndbox').find('ymax').text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():
    for folder in ['train', 'validation']:
        image_path = os.path.join(os.getcwd(), 'images', folder)
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(os.path.join('images', f'{folder}_labels.csv'), index=None)
        print(f'Successfully converted XML to CSV for folder: {folder}')

main()
