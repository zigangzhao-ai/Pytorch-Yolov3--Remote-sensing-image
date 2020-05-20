'''
code by zzg 2020-05-13
'''

#批量修改文件夹下的xml为json并存储到另一个文件夹

import glob
import xmltodict
import json

path = 'xml/'
path2 = 'json/'

xml_dir = glob.glob(path + '*.xml')
print(xml_dir)

def pythonXmlToJson(path):
  
    xml_dir = glob.glob(path + '*.xml')
    # print(len(xml_dir))
    for x in xml_dir:
        with open(x) as fd:
            convertedDict = xmltodict.parse(fd.read())
            jsonStr = json.dumps(convertedDict, indent=1)
            print("jsonStr=",jsonStr)
            print(x.split('.')[0])
            json_file = x.split('.')[0].split('/')[-1] +'.json'
            with open(path2 + '/' + json_file, 'w') as json_file:
                json_file.write(jsonStr)
    print("xml_json finished!")
    print(len(xml_dir))
pythonXmlToJson(path)
