'''
Author: Luis A. Gutiérrez-Soto
Script designed to perform iterative cross-matching of a table with SPLUS.cloud using unique IDs
'''
import pandas as pd
import os
import argparse
import logging
from pathlib import Path
from astropy.table import Table, vstack
import splusdata
from multiprocessing import Pool
from urllib3.exceptions import HeaderParsingError

logging.basicConfig(filename='script_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_to_splus(username, password):
    try:
        conn = splusdata.connect(username, password)
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to the SPLUS database: {str(e)}")
        return None

def query_chunk(query, conn, chunk):
    try:
        results = conn.query(query, chunk)
        if not results:
            logging.warning("Query returned no results.")
        return results
    except Exception as e:
        logging.error(f"Unexpected error while executing a SQL query: {str(e)}")
        return None

def read_data(input_file, batch_size):
    data = pd.read_csv(input_file)
    data['ID'] = data['ID'].str.replace(r"b'| '|       |'", "", regex=True)
    #data = data.iloc[0:5000]
    chunks = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    return chunks

def query_and_merge_chunks(query, conn, chunks):
    try:
        with Pool(processes=os.cpu_count()) as pool:
            results = pool.starmap(query_chunk, [(query, conn, chunk) for chunk in chunks])

        results = [res for res in results if res is not None]
        if results:
            merged_table = vstack(results)
            return merged_table
        else:
            return None

    except Exception as e:
        logging.error(f"Query and merge failed: {str(e)}")
        return None

def main():
    try:
        ROOT = Path("Final-list")
        parser = argparse.ArgumentParser(description="First table from the S-PLUS catalogs")
        parser.add_argument("table", type=str, default="teste-program", help="Name of the catalog, using the prefix")

        args = parser.parse_args()
        input_file = args.table + ".csv"

        batch_size = 3000

        logging.info(f"Processing data from {input_file}")

        checkpoint_file = f"{args.table}-checkpoint.txt"
        start_chunk = 0
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as checkpoint:
                start_chunk = int(checkpoint.read())

        username = os.environ.get('SPLUS_USERNAME')
        password = os.environ.get('SPLUS_PASSWORD')

        conn = connect_to_splus(username, password)
        if conn:
            chunks = read_data(input_file, batch_size)
            chunks_to_process = chunks[start_chunk:]

            query = """SELECT detection.Field, detection.ID, detection.RA, detection.DEC, detection.X, detection.Y,
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
		  LEFT OUTER JOIN idr4_dual.idr4_dual_u as u ON u.ID = detection.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_j0378 as J0378 ON J0378.ID = detection.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_j0395 as J0395 ON J0395.ID = detection.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_j0410 as J0410 ON J0410.ID = detection.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_j0430 as J0430 ON J0430.ID = detection.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_g as g ON g.ID = detection.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_j0515 as J0515 ON J0515.ID = detection.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_r as r ON r.ID = detection.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_j0660 as J0660 ON J0660.ID = detection.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_i as i ON i.ID = detection.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_j0861 as J0861 ON J0861.ID = detection.ID
		  LEFT OUTER JOIN idr4_dual.idr4_dual_z as z ON z.ID = detection.ID
                  LEFT OUTER JOIN idr4_psf.idr4_psf_u as u_psf ON u_psf.ID = detection.ID              
                  LEFT OUTER JOIN idr4_psf.idr4_psf_j0378 as J0378_psf ON J0378_psf.ID = detection.ID      
                  LEFT OUTER JOIN idr4_psf.idr4_psf_j0395 as J0395_psf ON J0395_psf.ID = detection.ID     
                  LEFT OUTER JOIN idr4_psf.idr4_psf_j0410 as J0410_psf ON J0410_psf.ID = detection.ID
                  LEFT OUTER JOIN idr4_psf.idr4_psf_j0430 as J0430_psf ON J0430_psf.ID = detection.ID          
                  LEFT OUTER JOIN idr4_psf.idr4_psf_g as g_psf ON g_psf.ID = detection.ID
                  LEFT OUTER JOIN idr4_psf.idr4_psf_j0515 as J0515_psf ON J0515_psf.ID = detection.ID      
                  LEFT OUTER JOIN idr4_psf.idr4_psf_r as r_psf ON r_psf.ID = detection.ID                 
                  LEFT OUTER JOIN idr4_psf.idr4_psf_j0660 as J0660_psf ON J0660_psf.ID = detection.ID     
                  LEFT OUTER JOIN idr4_psf.idr4_psf_i as i_psf ON i_psf.ID = detection.ID                
                  LEFT OUTER JOIN idr4_psf.idr4_psf_j0861 as J0861_psf ON J0861_psf.ID = detection.ID    
                  LEFT OUTER JOIN idr4_psf.idr4_psf_z as z_psf ON z_psf.ID = detection.ID
                  """

            merged_table = query_and_merge_chunks(query, conn, chunks_to_process)

            if merged_table:
                output_file = f"{args.table}-moreParameters.csv"
                merged_table.write(output_file, format='csv', overwrite=True)
                logging.info(f"Data processed and saved to {output_file}")
            else:
                logging.error("No valid results obtained.")

            with open(checkpoint_file, "w") as checkpoint:
                checkpoint.write(str(len(chunks_to_process) + start_chunk))

        else:
            logging.error("Connection to SPLUS failed.")

    except HeaderParsingError as hpe:
        logging.error(f"Header Parsing Error: {str(hpe)}")
    except Exception as e:
        logging.error(f"Script execution failed: {str(e)}")

if __name__ == "__main__":
    main()
