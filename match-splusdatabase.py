import splusdata
import getpass
import pandas as pd
import argparse
from astropy.table import Table, vstack
import numpy as np
from pathlib import Path
  
ROOT = Path("Final-list/") 
  
# Connecting with SPLUS database
username = str(input("Login: "))
password = getpass.getpass("Password: ")

conn = splusdata.connect(username, password)

#####################################################################
parser = argparse.ArgumentParser(
    description="""Firts table from the S-PLUS catalogs""")

parser.add_argument("table", type=str,
                    default=" teste-program",
                    help="Name of catalog, taken the prefix")

cmd_args = parser.parse_args()
file_ = cmd_args.table + ".csv"
                    
df =  pd.read_csv(file_)
#df_['ID'] = df_['ID'].str.decode('utf-8')
df['ID'] = df['ID'].str.replace(r"b'| '|       |'", "", regex=True)

print("Number of objects:", len(df))
  
Query = f"""SELECT detection.Field, detection.ID, detection.RA, detection.DEC, detection.X, detection.Y,
		  detection.FWHM,  detection.FWHM_n, detection.ISOarea, detection.KRON_RADIUS, 
		  detection.MU_MAX_INST, detection.PETRO_RADIUS, detection.SEX_FLAGS_DET, detection.SEX_NUMBER_DET,
                  detection.s2n_DET_PStotal, detection.THETA, 
		  u.u_PStotal, J0378.J0378_PStotal, J0395.J0395_PStotal,
		  J0410.J0410_PStotal, J0430.J0430_PStotal, g.g_PStotal,
		  J0515.J0515_PStotal, r.r_PStotal, J0660.J0660_PStotal, i.i_PStotal, 
		  J0861.J0861_PStotal, z.z_PStotal, u.e_u_PStotal, J0378.e_J0378_PStotal,
		  J0395.e_J0395_PStotal, J0410.e_J0410_PStotal, J0430.e_J0430_PStotal, 
		  g.e_g_PStotal, J0515.e_J0515_PStotal, r.e_r_PStotal, J0660.e_J0660_PStotal,
		  i.e_i_PStotal, J0861.e_J0861_PStotal, z.e_z_PStotal,
                  u_psf.u_psf, J0378_psf.J0378_psf, J0395_psf.J0395_psf,
		  J0410_psf.J0410_psf, J0430_psf.J0430_psf, g_psf.g_psf,
		  J0515_psf.J0515_psf, r_psf.r_psf, J0660_psf.J0660_psf, i_psf.i_psf, 
		  J0861_psf.J0861_psf, z_psf.z_psf, u_psf.e_u_psf, J0378_psf.e_J0378_psf,
		  J0395_psf.e_J0395_psf, J0410_psf.e_J0410_psf, J0430_psf.e_J0430_psf, 
		  g_psf.e_g_psf, J0515_psf.e_J0515_psf, r_psf.e_r_psf, J0660_psf.e_J0660_psf,
		  i_psf.e_i_psf, J0861_psf.e_J0861_psf, z_psf.e_z_psf
                  FROM TAP_UPLOAD.upload as tap 
                  LEFT OUTER JOIN idr4_dual.idr4_detection_image as detection ON (tap.ID = detection.ID) 
		  LEFT OUTER JOIN idr4_dual.idr4_dual_u as u ON tap.ID=u.ID 
		  LEFT OUTER JOIN idr4_dual.idr4_dual_j0378 as J0378 ON tap.ID=J0378.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_j0395 as J0395 ON tap.ID=J0395.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_j0410 as J0410 ON tap.ID=J0410.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_j0430 as J0430 ON tap.ID=J0430.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_g as g ON tap.ID=g.ID 
		  LEFT OUTER JOIN idr4_dual.idr4_dual_j0515 as J0515 ON tap.ID=J0515.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_r as r ON tap.ID=r.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_j0660 as J0660 ON tap.ID=J0660.ID 
		  LEFT OUTER JOIN idr4_dual.idr4_dual_i as i ON tap.ID=i.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_j0861 as J0861 ON tap.ID=J0861.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_z as z ON tap.ID=z.ID
                  LEFT OUTER JOIN idr4_psf.idr4_psf_u as u_psf ON tap.ID=u_psf.ID              
                  LEFT OUTER JOIN idr4_psf.idr4_psf_j0378 as J0378_psf ON tap.ID=J0378_psf.ID      
                  LEFT OUTER JOIN idr4_psf.idr4_psf_j0395 as J0395_psf ON tap.ID=J0395_psf.ID      
                  LEFT OUTER JOIN idr4_psf.idr4_psf_j0410 as J0410_psf ON tap.ID=J0410_psf.ID 
                  LEFT OUTER JOIN idr4_psf.idr4_psf_j0430 as J0430_psf ON tap.ID=J0430_psf.ID           
                  LEFT OUTER JOIN idr4_psf.idr4_psf_g as g_psf ON tap.ID=g_psf.ID                  
                  LEFT OUTER JOIN idr4_psf.idr4_psf_j0515 as J0515_psf ON tap.ID=J0515_psf.ID      
                  LEFT OUTER JOIN idr4_psf.idr4_psf_r as r_psf ON tap.ID=r_psf.ID                  
                  LEFT OUTER JOIN idr4_psf.idr4_psf_j0660 as J0660_psf ON tap.ID=J0660_psf.ID      
                  LEFT OUTER JOIN idr4_psf.idr4_psf_i as i_psf ON tap.ID=i_psf.ID                 
                  LEFT OUTER JOIN idr4_psf.idr4_psf_j0861 as J0861_psf ON tap.ID=J0861_psf.ID     
                  LEFT OUTER JOIN idr4_psf.idr4_psf_z as z_psf ON tap.ID=z_psf.ID
                  """               

# Count numbers of  tables done.
n = int(len(df) / 2000.) + 1
  
df_ = [] # list
j = 0 # counter
d = {} # empty
for i in range(n):
    j += 1  
    df_.append(df.iloc[2000*i:2000*j])
  
# Applying query
merged_table_list = []
for a in range(n):
    results = conn.query(Query, df_[a])
    merged_table_list.append(results)
  
# Merging all result astropy tables 
merged_table = vstack(merged_table_list)
print("Number objects with match:", len(merged_table))
  
# coverting in pandas table
df_merged = (merged_table.to_pandas())
df_merged.to_csv(file_.replace(".csv", "-moreParameters.csv"), index = False)
