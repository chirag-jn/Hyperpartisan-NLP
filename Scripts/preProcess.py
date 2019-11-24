from defaults import *
import xmlschema
import xml.etree.ElementTree as ET
from optparse import OptionParser

is_article = False

def setParser():
    parser = OptionParser()
    parser.add_option("--type", help="Article or Publisher", type=str, default='article')
    options, _ = parser.parse_args()
    return options

def validateXMLFiles():
    xmlschema.validate(article_training_data_loc, training_data_schema)
    xmlschema.validate(article_ground_truth_data_loc, ground_truth_schema)

def parseTrainingData():
    if is_article:
        tree = ET.parse(article_training_data_loc)
    else:
        tree = ET.parse(publisher_training_data_loc)
    root = tree.getroot()
    print(root.tag)
    print(root.attrib)
    for child in root:
        print(child.tag, child.attrib)
        print(len(child))
        break

def getGroundTruth():
    if is_article:
        tree = ET.parse(article_ground_truth_data_loc)
    else:
        tree = ET.parse(publisher_ground_truth_data_loc)
    root = tree.getroot()
    is_hyper = {}
    for child in root:
        attrib = dict(child.attrib)
        if attrib['hyperpartisan'] == 'true':
            is_hyper[attrib['id']] = True
        else:
            is_hyper[attrib['id']] = False
    return is_hyper

if __name__=='__main__':
    options = setParser()
    if options.type.lower() == 'article':
        is_article = True
    else:
        is_article = False
    validateXMLFiles()
    # print(getGroundTruth())
    parseTrainingData()