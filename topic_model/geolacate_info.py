import pandas as pd
from mordecai import Geoparser
from tqdm import tqdm 
import re 
from pycountry_convert import  country_alpha2_to_continent_code, country_alpha3_to_country_alpha2

geo = Geoparser()
#geo.geoparse("I took the tube from Oxford Circus to London Bridge, via Bank")
## should run in env mordecai

def get_loc_info( in_fp, out_fp ):
    data = pd.read_csv(in_fp, sep=",", index_col=False  ) 
    
    places = []
    geos = []
    for i, row in tqdm( data.iterrows()):  
        t = row['Title'].strip() + " " + row['Abstract'].strip()
        t = t.split("Copyright (C)")[0] 
        t = re.split("\([C-c]\) [1-2][0-9]{3} Elsevier",t)[0] 
        t = t.split("Published by Elsevier")[0] 
        t = t.split("Copyright. (C)")[0] 
        t = re.split("\. \(C\) [1-2][0-9]{3} ",t)[0] 
        t = re.split("\. \(C\) Copyright",t)[0]     
        if i == 57:
            import pdb; pdb.set_trace() 
        gp = geo.geoparse(t)
        rplaces = []
        continent = None
        add_geos = []
        for p in gp:
            try:
                a2 = country_alpha3_to_country_alpha2(p["country_predicted"])
                continent = country_alpha2_to_continent_code(a2)
            except:
                pass
            if "geo" in p:
                try:
                    a2 = country_alpha3_to_country_alpha2(p["geo"]["country_code3"])
                    continent = country_alpha2_to_continent_code(a2)
                except:
                    pass
                p["geo"]["doc_id"] = row['index']
                p["geo"]["word"] = p["word"]
                p["geo"]["country_predicted"] = p["country_predicted"]
                p["geo"]["country_conf"] = p["country_conf"]
                geos.append(p["geo"])
                add_geos.append(p["geo"])
            rplaces.append(p)
        """
        if len(add_geos) == 0:
            print(t)
            import  pdb; pdb.set_trace() 
        """
        data.loc[i,"continent"] = continent
        data.loc[i,"places"] = len(rplaces) 
    
       
    geo_df = pd.DataFrame.from_dict(geos)
    geo_df.to_csv(out_fp, sep=",", index=False)
    geo_df.head()
    return

folder="data folder"

in_fp = folder + "/valid_data.csv" 
out_fp = folder + "/data_loc.csv" 
get_loc_info( in_fp, out_fp )
## runs about 1.5h 