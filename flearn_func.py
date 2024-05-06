
def flearn_pp(df,feat_eng=True):
    import numpy as np
    df.drop_duplicates("patient_nbr",inplace=True)
    df=df[~df["discharge_disposition_id"].isin([11,13,14,19,20,21])].copy()
    df=df[df["gender"]!="Unknown/Invalid"].copy()

    # Create Outcome variables
    
    
    # Replace missing values and re-code categories
    df.loc[:,"age"] = df.age.replace({"?": ""})
    df.loc[:,"payer_code"] = df["payer_code"].replace({"?", "Unknown"})
    df.loc[:,"medical_specialty"] = df["medical_specialty"].replace({"?": "Missing"})
    df.loc[:, "race"] = df["race"].replace({"?": "Unknown"})
    df["age_int"]=df["age"].map({'[0-10)':5, '[10-20)':15, '[20-30)':25, '[30-40)':35, '[40-50)':45, '[50-60)':55, '[60-70)':65, '[70-80)':75, '[80-90)':85, '[90-100)':95})#

    df.loc[:, "admission_source_id"] = df["admission_source_id"].replace({1: "Referral", 2: "Referral", 3: "Referral", 7: "Emergency"})
    df.loc[:, "age"] = df["age"].replace( ["[0-10)", "[10-20)", "[20-30)"], "30 years or younger")
    df.loc[:, "age"] = df["age"].replace(["[30-40)", "[40-50)", "[50-60)"], "30-60 years")
    df.loc[:, "age"] = df["age"].replace(["[60-70)", "[70-80)", "[80-90)","[90-100)"], "Over 60 years")

    # Clean various medical codes
    df.loc[:, "discharge_disposition_id"] = (df.discharge_disposition_id
                                            .apply(lambda x: "Discharged to Home" if x==1 else "Other"))

    df.loc[:, "admission_source_id"] = df["admission_source_id"].apply(lambda x: x if x in ["Emergency", "Referral"] else "Other")
    # Re-code Medical Specialties and Primary Diagnosis
    specialties = [
        "Missing",
        "InternalMedicine",
        "Emergency/Trauma",
        "Family/GeneralPractice",
        "Cardiology",
        "Surgery"
    ]
    df.loc[:, "medical_specialty"] = df["medical_specialty"].apply(lambda x: x if x in specialties else "Other")
    #
    df.loc[:, "diag_1"] = df["diag_1"].replace(
        regex={
            "[7][1-3][0-9]": "Musculoskeltal Issues",
            "250.*": "Diabetes",
            "[4][6-9][0-9]|[5][0-1][0-9]|786": "Respitory Issues",
            "[5][8-9][0-9]|[6][0-2][0-9]|788": "Genitourinary Issues"
        }
    )
    df.loc[:, "diag_2"] = df["diag_2"].replace(
        regex={
            "[7][1-3][0-9]": "Musculoskeltal Issues",
            "250.*": "Diabetes",
            "[4][6-9][0-9]|[5][0-1][0-9]|786": "Respitory Issues",
            "[5][8-9][0-9]|[6][0-2][0-9]|788": "Genitourinary Issues"
        }
    )
    df.loc[:, "diag_3"] = df["diag_3"].replace(
        regex={
            "[7][1-3][0-9]": "Musculoskeltal Issues",
            "250.*": "Diabetes",
            "[4][6-9][0-9]|[5][0-1][0-9]|786": "Respitory Issues",
            "[5][8-9][0-9]|[6][0-2][0-9]|788": "Genitourinary Issues"
        }
    )
    diagnoses = ["Respitory Issues", "Diabetes", "Genitourinary Issues", "Musculoskeltal Issues"]
    df.loc[:, "diag_1"] = df["diag_1"].apply(lambda x: x if x in diagnoses else "Other")
    df.loc[:, "diag_2"] = df["diag_2"].apply(lambda x: x if x in diagnoses else "Other")
    df.loc[:, "diag_3"] = df["diag_3"].apply(lambda x: x if x in diagnoses else "Other")
    num_cols=['age_int',  'time_in_hospital',
       'num_lab_procedures', 'num_procedures', 'num_medications','number_diagnoses',
       'change', 'diabetesMed', 'medicare', 'medicaid',
       'had_emergency', 'had_inpatient_days', 'had_outpatient_days']
    cols_to_keep = ["race","gender","age","age_int","discharge_disposition_id","admission_source_id","time_in_hospital",
        "num_lab_procedures","num_procedures","num_medications","diag_1","diag_2","diag_3","number_diagnoses","medical_specialty",
        "max_glu_serum","A1Cresult","insulin","change",
        "diabetesMed", "medicare", "medicaid", "had_emergency", "had_inpatient_days", "had_outpatient_days","readmitted",]

    if feat_eng == True:
        print("Feature Engineering...")
        ########eli feature engineering https://ieeexplore.ieee.org/document/9035280 https://link.springer.com/chapter/10.1007/978-981-15-5345-5_7#Sec3
        #add med change
        med_df=df[['metformin',
        'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
        'tolazamide', 'examide', 'citoglipton', 'insulin',
        'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone',]].copy()

        df["med_abs_change"]=med_df.replace({"Up":1,"Down":1,"Steady":0,"No":0}).sum(1)
        df["med_up_change"]=med_df.replace({"Up":1,"Down":0,"Steady":0,"No":0}).sum(1)
        df["med_down_change"]=med_df.replace({"Up":0,"Down":1,"Steady":0,"No":0}).sum(1)


        df["healthcare_service_total_visits"]=np.log1p(df[["number_emergency","number_inpatient","number_outpatient"]]).sum(1)
        def time_in_hos_bin(time_col):
            bin=[]
            for i in time_col:
                if i < 5:
                    bin.append("short")
                elif i < 10:
                    bin.append("medium")
                else:
                    bin.append("long")
            return bin
        df.loc[:,"time_hos_bin"]=  time_in_hos_bin(df["time_in_hospital"])  

        num_cols=['age_int',  'time_in_hospital',
       'num_lab_procedures', 'num_procedures', 'num_medications','number_diagnoses',
       'change', 'diabetesMed', 'medicare', 'medicaid',"med_abs_change","med_up_change","med_down_change","healthcare_service_total_visits",
       'had_emergency', 'had_inpatient_days', 'had_outpatient_days']
        cols_to_keep = ["race","gender","age","age_int","discharge_disposition_id","admission_source_id","time_in_hospital",
        "num_lab_procedures","num_procedures","num_medications","diag_1","diag_2","diag_3","number_diagnoses","medical_specialty",
                    "max_glu_serum","A1Cresult","insulin","change","med_abs_change","med_up_change","med_down_change","healthcare_service_total_visits","time_hos_bin",
        "diabetesMed", "medicare", "medicaid", "had_emergency", "had_inpatient_days", "had_outpatient_days","readmitted",]

        


    #Binarize and bin features
    df.loc[:, "medicare"] = (df.payer_code == "MC")
    df.loc[:, "medicaid"] = (df.payer_code == "MD")

    df.loc[:, "had_emergency"] = (df["number_emergency"] > 0)
    df.loc[:, "had_inpatient_days"] = (df["number_inpatient"] > 0)
    df.loc[:, "had_outpatient_days"] = (df["number_outpatient"] > 0)
    df["change"]=df["change"]!="No"
    df["diabetesMed"]=df["diabetesMed"]!="No"
    df["readmitted"]=df["readmitted"]=="<30"
    df["max_glu_serum"]=[None if i=="nan" else i for i in df["max_glu_serum"]]
    df["A1Cresult"]=[None if i=="nan" else i for i in df["A1Cresult"]]

    df[num_cols]=df[num_cols].astype(int)
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols]=df[cat_cols].astype("category")
    # Save DataFrame

    final_df = df.loc[:, cols_to_keep]

    return final_df
    #final_df.to_csv(data_path / "diabetic_preprocessed.csv", index=False)

