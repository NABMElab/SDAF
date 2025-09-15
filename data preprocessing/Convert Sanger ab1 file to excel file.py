from Bio import SeqIO
import matplotlib.pyplot as plt
import pandas as pd

def sequence(file_name):
    info_dict = {}
    raw = open(file_name, errors='ignore').read()
    if file_name[-3:] != 'ab1' or raw[:4] != 'ABIF':
        return "wrong file format"

    for record in SeqIO.parse(file_name, "abi"):
        info_dict["seq"] = record.seq
        info_dict["name"] = record.id
        anno = record.annotations
        letter_anno = record.letter_annotations
        abif_raw = anno["abif_raw"]
        info_dict["date"] = anno["run_start"] + " to " + anno["run_finish"]
        # info_dict["lane"] = anno["LANE1"]
        info_dict["spac"] = "{:.2f}".format(abif_raw["SPAC1"])
        info_dict["dyep"] = abif_raw["PDMF2"].decode('utf-8')
        info_dict["mach"] = abif_raw["MCHN1"].decode('utf-8')
        info_dict["modl"] = anno["machine_model"].decode('utf-8') 
        info_dict["bcal"] = abif_raw["SPAC2"].decode('utf-8')
        info_dict["ver1"] = abif_raw["SVER1"].decode('utf-8')
        info_dict["ver2"] = abif_raw["SVER2"].decode('utf-8')

        data_c = list(abif_raw["DATA9"])
        data_t = list(abif_raw["DATA10"])
        data_a = list(abif_raw["DATA11"])
        data_g = list(abif_raw["DATA12"])

        qs = letter_anno["phred_quality"]

        for k, v in info_dict.items():
            print(k + " : " + v)
        print("qs:")
        print(qs)

        data = [data_a,data_g,data_c,data_t]
        seq_map = {0:'A', 1:'G', 2:'C', 3:'T'}
        winner1 = 1
        result = []
        result_seq = []
        for i in range(len(data_a)):
            if(i < 0 or i > (len(data_a) - 3) ):
                continue
            else:
                temp = data[winner1][i]
                result.append(temp)
                result_seq.append(seq_map[winner1])

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)
        print(df)

        # Save the DataFrame to an Excel file without index
        excel_file = "C:/Users/Jack Leong/Downloads/20250702_补充组织解谱/20250702_tube2c-7P-WB-0.5XHB-250112-S21.xlsx"  # Define the output Excel file path
        df.to_excel(excel_file, index=False)  

# Input Sanger ab1 file to convert
sequence("C:/Users/Jack Leong/Downloads/20250702_补充组织解谱/tube2C-7p-0_5X-HB-200nM-WB-250112-S21.AD.20713951.119726A02.A12.ab1")

