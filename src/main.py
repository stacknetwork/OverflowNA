
import pandas as pd
import networkx as nx
import xml.etree.ElementTree as ET
import numpy as np

def iter_docs(author):
  print(author)
  author_attr = author.attrib
  for doc in author.iter('document'):
    doc_dict = author_attr.copy()
    doc_dict.update(doc.attrib)
    doc_dict['data'] = doc.text
    yield doc_dict
        

if __name__ == "__main__":
  print("parsing")
  etree = ET.parse("./data/stackoverflow.com-Badges/Badges.xml")
  print(etree)
  doc_df = pd.DataFrame(list(iter_docs(etree.getroot())))
