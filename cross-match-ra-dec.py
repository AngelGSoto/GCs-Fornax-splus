import concurrent.futures
import pandas as pd
import splusdata
import argparse
import os
import logging
import time

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_environment_variables():
    username = os.environ.get('SPLUS_USERNAME')
    password = os.environ.get('SPLUS_PASSWORD')
    
    if not (username and password):
        raise ValueError("SPLUS_USERNAME and SPLUS_PASSWORD environment variables are required.")

def process_field(field, query_template):
    logging.info(f"Running for: {field}")
    query = query_template.format(field=field)

    username = os.environ.get('SPLUS_USERNAME')
    password = os.environ.get('SPLUS_PASSWORD')

    conn = splusdata.connect(username, password)

    for retry in range(MAX_RETRIES):
        try:
            tab = conn.query(query, df).to_pandas()
            logging.info(f"Found: {len(tab)} objects for {field}")
            return tab
        except IndexError as ie:
            logging.error(f"IndexError in query for field {field}: {ie}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error in query for field {field}: {e}")
            if retry < MAX_RETRIES - 1:
                logging.info(f"Retrying request for {field} after {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logging.error(f"All retries exhausted for field {field}.")
                return pd.DataFrame()

def main(test_mode=False):
    setup_logging()
    check_environment_variables()
    
    query_template = """
        SELECT 
            TOP 5 det.Field, det.ID, det.RA, det.DEC, det.X, det.Y,
            det.FWHM,  det.FWHM_n, det.ISOarea, det.KRON_RADIUS, 
            det.MU_MAX_INST, det.PETRO_RADIUS, det.SEX_FLAGS_DET, det.SEX_NUMBER_DET, det.CLASS_STAR,
            det.s2n_DET_PStotal, det.THETA, det.ELLIPTICITY, det.ELONGATION,
            det.FLUX_RADIUS_20, det.FLUX_RADIUS_50, det.FLUX_RADIUS_70, det.FLUX_RADIUS_90,  
            r.s2n_r_PStotal, j0660.s2n_J0660_PStotal, i.s2n_i_PStotal,
            r.FWHM_r, r.FWHM_n_r, j0660.FWHM_J0660, j0660.FWHM_n_J0660, i.FWHM_i, i.FWHM_n_i,
            r.SEX_FLAGS_r, j0660.SEX_FLAGS_J0660, i.SEX_FLAGS_i,
            rs.CLASS_STAR_r, j0660s.CLASS_STAR_J0660, isf.CLASS_STAR_i,
        
            r.r_PStotal, r.e_r_PStotal,
            g.g_PStotal, g.e_g_PStotal,
            i.i_PStotal, i.e_i_PStotal,
            u.u_PStotal, u.e_u_PStotal,
            z.z_PStotal, z.e_z_PStotal,
            j0378.j0378_PStotal, j0378.e_j0378_PStotal,
            j0395.j0395_PStotal, j0395.e_j0395_PStotal,
            j0410.j0410_PStotal, j0410.e_j0410_PStotal,
            j0430.j0430_PStotal, j0430.e_j0430_PStotal,
            j0515.j0515_PStotal, j0515.e_j0515_PStotal,
            j0660.j0660_PStotal, j0660.e_j0660_PStotal,
            j0861.j0861_PStotal, j0861.e_j0861_PStotal,
        
            rs.r_psf, rs.e_r_psf,
            gs.g_psf, gs.e_g_psf,
            isf.i_psf, isf.e_i_psf, -- isf (is -> reserved)
            us.u_psf, us.e_u_psf,
            zs.z_psf, zs.e_z_psf,
            j0378s.j0378_psf, j0378s.e_j0378_psf,
            j0395s.j0395_psf, j0395s.e_j0395_psf,
            j0410s.j0410_psf, j0410s.e_j0410_psf,
            j0430s.j0430_psf, j0430s.e_j0430_psf,
            j0515s.j0515_psf, j0515s.e_j0515_psf,
            j0660s.j0660_psf, j0660s.e_j0660_psf,
            j0861s.j0861_psf, j0861s.e_j0861_psf
        
        FROM TAP_UPLOAD.upload AS tap
        LEFT OUTER JOIN idr4_dual.idr4_detection_image AS det 
            ON (1=CONTAINS(POINT('ICRS', det.RA, det.DEC), 
                CIRCLE('ICRS', tap.RA, tap.DEC, 0.000277777777778)))
        LEFT OUTER JOIN idr4_dual.idr4_dual_r AS r ON r.id = det.id
        LEFT OUTER JOIN idr4_dual.idr4_dual_g AS g ON g.id = det.id
        LEFT OUTER JOIN idr4_dual.idr4_dual_i AS i ON i.id = det.id
        LEFT OUTER JOIN idr4_dual.idr4_dual_z AS z ON z.id = det.id
        LEFT OUTER JOIN idr4_dual.idr4_dual_u AS u ON u.id = det.id
        
        LEFT OUTER JOIN idr4_dual.idr4_dual_j0378 AS j0378 ON j0378.id = det.id
        LEFT OUTER JOIN idr4_dual.idr4_dual_j0395 AS j0395 ON j0395.id = det.id
        LEFT OUTER JOIN idr4_dual.idr4_dual_j0410 AS j0410 ON j0410.id = det.id
        LEFT OUTER JOIN idr4_dual.idr4_dual_j0430 AS j0430 ON j0430.id = det.id
        LEFT OUTER JOIN idr4_dual.idr4_dual_j0515 AS j0515 ON j0515.id = det.id
        LEFT OUTER JOIN idr4_dual.idr4_dual_j0660 AS j0660 ON j0660.id = det.id
        LEFT OUTER JOIN idr4_dual.idr4_dual_j0861 AS j0861 ON j0861.id = det.id
        
        LEFT OUTER JOIN idr4_psf.idr4_psf_r AS rs ON rs.id = det.id
        LEFT OUTER JOIN idr4_psf.idr4_psf_g AS gs ON gs.id = det.id
        LEFT OUTER JOIN idr4_psf.idr4_psf_i AS isf ON isf.id = det.id
        LEFT OUTER JOIN idr4_psf.idr4_psf_z AS zs ON zs.id = det.id
        LEFT OUTER JOIN idr4_psf.idr4_psf_u AS us ON us.id = det.id
        
        LEFT OUTER JOIN idr4_psf.idr4_psf_j0378 AS j0378s ON j0378s.id = det.id
        LEFT OUTER JOIN idr4_psf.idr4_psf_j0395 AS j0395s ON j0395s.id = det.id
        LEFT OUTER JOIN idr4_psf.idr4_psf_j0410 AS j0410s ON j0410s.id = det.id
        LEFT OUTER JOIN idr4_psf.idr4_psf_j0430 AS j0430s ON j0430s.id = det.id
        LEFT OUTER JOIN idr4_psf.idr4_psf_j0515 AS j0515s ON j0515s.id = det.id
        LEFT OUTER JOIN idr4_psf.idr4_psf_j0660 AS j0660s ON j0660s.id = det.id
        LEFT OUTER JOIN idr4_psf.idr4_psf_j0861 AS j0861s ON j0861s.id = det.id
              
        WHERE 
            det.field = '{field}'
    """

    fields = pd.read_csv("https://splus.cloud/files/documentation/iDR4/tabelas/iDR4_zero-points.csv")
    
    LIMIT_OBJECTS = 2000
    final_table = pd.DataFrame()

    for field in fields["Field"] if not test_mode else fields["Field"][:5]:
        result = process_field(field, query_template)
        final_table = pd.concat([final_table, result])

    final_table.to_csv(f"iDR4-SPLUS-PStotal-PSF-185r195_class05_flags4.csv", index=False)

if __name__ == "__main__":
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds

    test_mode = False  # Set this to False when you are ready to run on the entire dataset
    parser = argparse.ArgumentParser(description="""First table from the S-PLUS catalogs""")

    parser.add_argument("table", type=str, default="teste-program", help="Name of catalog, taken the prefix")
    cmd_args = parser.parse_args()
    file_ = cmd_args.table + ".csv"

    df = pd.read_csv(file_)
    main(test_mode)
