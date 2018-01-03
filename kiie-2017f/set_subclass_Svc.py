# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 21:20:35 2017

@author: Hanbin Seo
"""

import requests
import bs4
import os
import json

#from set_subclass_Mfr import print_count_subcls_list

def get_IPC_from_USPC(digit):
    dict_uspc2ipc, cutoff = dict(), digit-6 # ex. 4 digit = -2
    for tr in tr_arr[1:]:
        sector, class_, sub_class = tr.find_all("td")[1].text.replace("\xa0", "").strip().split()
        group, _, sub_group = tr.find_all("td")[2].text.replace("\xa0", "").strip().split()
        res_tmp = [sector, class_, sub_class, group, sub_group]
        res = tuple(res_tmp[:cutoff])
        if res not in dict_uspc2ipc.keys():
            dict_uspc2ipc[res] = 1
        else:
            dict_uspc2ipc[res] += 1
    print("len: {}".format(len(dict_uspc2ipc.keys())))
    return dict_uspc2ipc

def cov_digit_set(dict_uspc2ipc):
    digit_set = [''.join(key) for key in dict_uspc2ipc.keys()]
    #digit_set = tuple(digit_set)
    return digit_set


if __name__ == '__main__':

    base_url = "https://www.uspto.gov/web/patents/classification/uspc705/us705toipc8.htm"
    res = requests.get(base_url)
    bsoup = bs4.BeautifulSoup(res.text, "lxml")
    bsoup = bsoup.body
    tr_arr = bsoup.find_all("tr", valign="middle", align="center")

    dict_subcls_list = {
            "subclass_Svc.": None, "subclass_Mfr.": None, "subclass_Tot.": None
            }
    dict_subcls_list['subclass_Svc.'] = cov_digit_set(get_IPC_from_USPC(4))
    print(dict_subcls_list)
    
    
    print(get_IPC_from_USPC(4)) 
 """{('A', '01', 'K'): 1,
     ('B', '65', 'B'): 1,
     ('G', '01', 'G'): 2,
     ('G', '01', 'R'): 2,
     ('G', '06', 'F'): 20, # ELECTRIC DIGITAL DATA PROCESSING (computer systems based on specific computational models G06N) 
     ('G', '06', 'G'): 3,
     ('G', '06', 'Q'): 38, # DATA PROCESSING SYSTEMS OR METHODS, SPECIALLY ADAPTED FOR ADMINISTRATIVE, COMMERCIAL, FINANCIAL, MANAGERIAL, SUPERVISORY OR FORECASTING PURPOSES; SYSTEMS OR METHODS SPECIALLY ADAPTED FOR ADMINISTRATIVE, COMMERCIAL, FINANCIAL, MANAGERIAL, SUPERVISORY OR FORECASTING PURPOSES, NOT OTHERWISE PROVIDED FOR [2006.01]
     ('G', '07', 'B'): 12, # TICKET-ISSUING APPARATUS; TAXIMETERS; ARRANGEMENTS OR APPARATUS FOR COLLECTING FARES, TOLLS OR ENTRANCE FEES AT ONE OR MORE CONTROL POINTS; FRANKING APPARATUS
     ('G', '07', 'C'): 2,
     ('G', '07', 'F'): 5,
     ('G', '07', 'G'): 4,
     ('H', '04', 'M'): 1}"""
    
    os.chdir("D:\Seminar_2017_summer\Paper_fall")
    WORK_DIR = os.getcwd().replace('\\','/')
    
    with open(WORK_DIR+"/data_json/subclass_list_ini.json", 'w') as f:
        json.dump(dict_subcls_list, f)
