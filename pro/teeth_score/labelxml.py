# coding: utf-8
# 基于lxml, 安装:pip install lxml

from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import lxml.etree as etree
import pprint
import os


def main():
    create_xml(filename='None')


def indent(elem, level=0):
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def create_xml(xml_name='None.xml', path='None', labelname='None', width='0', height='0', xmin='0', ymin='0', xmax='0', ymax='0'):
    node_root = Element('annotation')
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = os.path.basename(path)
    node_path = SubElement(node_root, 'path')
    node_path.text = path
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = width
    node_height = SubElement(node_size, 'height')
    node_height.text = height
    node_bndbox = SubElement(node_root, 'bndbox')
    node_xmin = SubElement(node_bndbox, 'xmin')
    node_xmin.text = xmin
    node_ymin = SubElement(node_bndbox, 'ymin')
    node_ymin.text = ymin
    node_xmax = SubElement(node_bndbox, 'xmax')
    node_xmax.text = xmax
    node_ymax = SubElement(node_bndbox, 'ymax')
    node_ymax.text = ymax

    # print etree.tostring(node_root, pretty_print=True)

    indent(node_root)
    tree = etree.ElementTree(node_root)
    tree.write(xml_name, pretty_print=True,
               xml_declaration=False, encoding='utf-8')


if __name__ == "__main__":
    main()
