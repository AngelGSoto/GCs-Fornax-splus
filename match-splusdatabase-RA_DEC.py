import splusdata
import getpass
import pandas as pd
import argparse
from astropy.table import Table, vstack
from pathlib import Path

ROOT = Path("Final-list/") 

# Connecting with SPLUS database
username = str(input("Login: "))
password = getpass.getpass("Password: ")

conn = splusdata.connect(username, password)

#####################################################################
parser = argparse.ArgumentParser(
    description="""First table from the S-PLUS catalogs""")

parser.add_argument("table", type=str,
                    default="teste-program",
                    help="Name of catalog, taken the prefix")

cmd_args = parser.parse_args()
file_ = cmd_args.table + ".csv"

df = pd.read_csv(file_)

print("Number of objects:", len(df))

# Query to perform cross-match using RA and DEC
Query = """
SELECT detection.Field, detection.ID, detection.RA, detection.DEC, detection.X, detection.Y,
    detection.FWHM, detection.FWHM_n, detection.ISOarea, detection.KRON_RADIUS, 
    detection.MU_MAX_INST, detection.PETRO_RADIUS, detection.SEX_FLAGS_DET, detection.SEX_NUMBER_DET,
    detection.s2n_DET_PStotal, detection.THETA
FROM TAP_UPLOAD.upload AS tap
LEFT OUTER JOIN idr4_dual.idr4_detection_image AS detection 
    ON (1=CONTAINS( POINT('ICRS', detection.RA, detection.DEC), 
        CIRCLE('ICRS', tap.RA, tap.DEC, 0.000277777777778)))
"""

# Count numbers of tables done.
n = int(len(df) / 3000.) + 1
print('n', n)  
df_ = [] # list
j = 0 # counter
d = {} # empty
for i in range(n):
    j += 1  
    df_.append(df.iloc[3000*i:3000*j])

# Applying query
merged_table_list = []

for a in range(n):
    try:
        results = conn.query(Query, df_[a])
        
        if isinstance(results, Table):
            merged_table_list.append(results)
        elif isinstance(results, pd.DataFrame):
            # Convert DataFrame to Astropy Table
            astropy_table = Table.from_pandas(results)
            merged_table_list.append(astropy_table)
        else:
            print(f"Results for chunk {a} is not of type Table or DataFrame. Type: {type(results)}")
            
    except Exception as e:
        print(f"Error occurred while querying chunk {a}: {e}")

# Merging all result astropy tables 
merged_table = vstack(merged_table_list)
print("Number objects with match:", len(merged_table))

# converting to pandas table
df_merged = merged_table.to_pandas()
df_merged.to_csv(file_.replace(".csv", "-SPLUS_ID.csv"), index=False)
