
import pandas as pd
import networkx as nx
from lxml import etree
import numpy as np
from datetime import datetime

def parse_users(min_date, max_date):
  users_xml_items = ["Id", "Reputation", "CreationDate",
   "DisplayName", "LastAccessDate", "WebsiteUrl", "Location", "AboutMe",
    "Views", "UpVotes", "DownVotes", "ProfileImageUrl", "AccountId"]

  maxtime = datetime.strptime(max_date, "%Y-%m-%dT%H:%M:%S.%f")
  mintime = datetime.strptime(min_date, "%Y-%m-%dT%H:%M:%S.%f")

  df = pd.DataFrame(columns=users_xml_items)
  i= 0
  rows = list()
  for event, element in etree.iterparse("./data/Users.xml"):
    if i % 1000000 == 0:
      print(i,len(rows))
    try:
      time = datetime.strptime(element.items()[2][1], "%Y-%m-%dT%H:%M:%S.%f")
      if ( time > mintime  and time < maxtime):
        rows.append(element.items())
    except:
      print(element.items())
    i = i + 1
    element.clear()

  df = pd.DataFrame([[v for k,v in r] for r in rows], columns=users_xml_items)
  df.to_pickle("data/Users.pkl")
  return df
        

if __name__ == "__main__":
  print("parsing")
  #df = pd.read_pickle("data/Users.pkl")
  #print(df)
  parse_users('2014-01-01T00:00:00.000', '2016-01-01T00:00:00.000')
