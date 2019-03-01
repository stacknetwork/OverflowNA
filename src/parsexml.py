
import pandas as pd
import networkx as nx
from lxml import etree
import numpy as np
from datetime import datetime

def parse_xml(xml, min_date, max_date):


  maxtime = datetime.strptime(max_date, "%Y-%m-%dT%H:%M:%S.%f")
  mintime = datetime.strptime(min_date, "%Y-%m-%dT%H:%M:%S.%f")

  i= 0
  rows = list()
  for event, element in etree.iterparse(xml):
   # if i % 100000 == 0:
     # print(i,len(rows))
   # try:
      #time = datetime.strptime(dict(element.items())["CreationDate"], "%Y-%m-%dT%H:%M:%S.%f")
     # if ( time > mintime  and time < maxtime):
      #  rows.append(element.items())
    #except:
     # print(element.items())
    i = i + 1
    element.clear()
  print(i)
 # df = pd.DataFrame.from_records([{k: v for k, v in row} for row in rows])
 # df.to_pickle(xml.split(".")[0]+"week_2.pkl")
  return df


if __name__ == "__main__":
  print("parsing")
  # df = pd.read_pickle("data/Users.pkl")
  # print(df)
  
  parse_xml("data/Users.xml", '2015-10-01T00:00:00.000', '2015-10-07T00:00:00.000')
  parse_xml("data/Comments.xml", '2015-10-01T00:00:00.000', '2015-10-07T00:00:00.000')
  parse_xml("data/Posts.xml", '2015-10-01T00:00:00.000', '2015-10-07T00:00:00.000')