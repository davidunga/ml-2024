def filtering_and_train_test_split(df,test_size=0.2,random_state=42):

    """Input: raw dataframe
       Output: X_train and X_test dataframes + y_test and y_train"""

    import numpy as np
    from sklearn.model_selection import train_test_split

    df.drop_duplicates("patient_nbr",inplace=True)
    df=df[~df["discharge_disposition_id"].isin([11,13,14,19,20,21])].copy()
    df=df[df["gender"]!="Unknown/Invalid"].copy()
    df["readmitted"]=df["readmitted"]=="<30"
    print("df.shape =",df.shape)
    print("df.readmitted.value_counts:",df.readmitted.value_counts())

    y=df.readmitted
    df=df.iloc[:,df.columns!="readmitted"]

    return train_test_split(df, y,stratify=y, test_size=test_size, random_state=random_state)

        
    
def InitialPP_func(df,return_y=False ,feat_eng=True,sqrt=True,):
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        import numpy as np
        df.drop_duplicates("patient_nbr",inplace=True)
        df=df[~df["discharge_disposition_id"].isin([11,13,14,19,20,21])].copy()
        df=df[df["gender"]!="Unknown/Invalid"].copy()
        

        # Create Outcome variables


        # Replace missing values and re-code categories
        df.loc[:,"age"] = df.age.replace({"?": ""})
        df.loc[:,"payer_code"] = df["payer_code"].replace({"?": "Unknown"})
        df.loc[:,"medical_specialty"] = df["medical_specialty"].replace({"?": "Missing"})
        df.loc[:, "race"] = df["race"].replace({"?": "Unknown"})
        df["age_int"]=df["age"].replace({'[0-10)':5, '[10-20)':15, '[20-30)':25, '[30-40)':35, '[40-50)':45, '[50-60)':55, '[60-70)':65, '[70-80)':75, '[80-90)':85, '[90-100)':95})#

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


        num_cols=[ "age_int",'time_in_hospital','num_lab_procedures', 'num_procedures', 'num_medications',
                      'number_diagnoses',"number_emergency","number_inpatient","number_outpatient",
                        #"med_abs_change","med_up_change","med_down_change","healthcare_service_total_visits",
                        ]
        bool_cols=["change","diabetesMed","medicare","medicaid","had_emergency","had_inpatient_days","had_outpatient_days"]
        cols_to_keep = ["race","gender","age_int","discharge_disposition_id","admission_source_id","time_in_hospital",
                            "number_emergency","number_inpatient","number_outpatient",
                                "num_lab_procedures","num_procedures","num_medications","diag_1","diag_2","diag_3","number_diagnoses",
                                "medical_specialty",
                        #"max_glu_serum","A1Cresult",
                        "insulin","change",
                        "max_glu_serum_","A1Cresult_",
                       # "max_glu_serum__","A1Cresult__",
                        #"med_abs_change","med_up_change","med_down_change","healthcare_service_total_visits","time_hos_bin",
            "diabetesMed",
             # "medicare", "medicaid",
            "medicare/medicaid",
              #"had_emergency", "had_inpatient_days", "had_outpatient_days","age"
              #"med_biguanides_and_related_abs_change","med_biguanides_and_related_up_change","med_biguanides_and_related_down_change","med_sulfonylureas_and_meglitinides_abs_change","med_sulfonylureas_and_meglitinides_up_change","med_sulfonylureas_and_meglitinides_down_change","med_thiazolidinediones_and_miscellaneous_abs_change","med_thiazolidinediones_and_miscellaneous_up_change","med_thiazolidinediones_and_miscellaneous_down_change",
              ]


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


            drugs_dict = {
        "biguanides_and_related": ["metformin", "glyburide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone"],
    "sulfonylureas_and_meglitinides": ["repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide"],
    "thiazolidinediones_and_miscellaneous": ["pioglitazone", "rosiglitazone", "troglitazone", "acarbose", "miglitol", "tolazamide", "examide", "citoglipton", "glipizide-metformin", "metformin-pioglitazone"]
}

            df["med_biguanides_and_related_abs_change"]=med_df[drugs_dict["biguanides_and_related"]].replace({"Up":1,"Down":1,"Steady":0,"No":0}).sum(1)
            df["med_biguanides_and_related_up_change"]=med_df[drugs_dict["biguanides_and_related"]].replace({"Up":1,"Down":0,"Steady":0,"No":0}).sum(1)
            df["med_biguanides_and_related_down_change"]=med_df[drugs_dict["biguanides_and_related"]].replace({"Up":0,"Down":1,"Steady":0,"No":0}).sum(1)
        
            df["med_sulfonylureas_and_meglitinides_abs_change"]=med_df[drugs_dict["sulfonylureas_and_meglitinides"]].replace({"Up":1,"Down":1,"Steady":0,"No":0}).sum(1)
            df["med_sulfonylureas_and_meglitinides_up_change"]=med_df[drugs_dict["sulfonylureas_and_meglitinides"]].replace({"Up":1,"Down":0,"Steady":0,"No":0}).sum(1)
            df["med_sulfonylureas_and_meglitinides_down_change"]=med_df[drugs_dict["sulfonylureas_and_meglitinides"]].replace({"Up":0,"Down":1,"Steady":0,"No":0}).sum(1)
            df["med_thiazolidinediones_and_miscellaneous_abs_change"]=med_df[drugs_dict["thiazolidinediones_and_miscellaneous"]].replace({"Up":1,"Down":1,"Steady":0,"No":0}).sum(1)
            df["med_thiazolidinediones_and_miscellaneous_up_change"]=med_df[drugs_dict["thiazolidinediones_and_miscellaneous"]].replace({"Up":1,"Down":0,"Steady":0,"No":0}).sum(1)
            df["med_thiazolidinediones_and_miscellaneous_down_change"]=med_df[drugs_dict["thiazolidinediones_and_miscellaneous"]].replace({"Up":0,"Down":1,"Steady":0,"No":0}).sum(1)







            df["healthcare_service_total_visits"]=(df[["number_emergency","number_inpatient","number_outpatient"]]).sum(1)
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

            num_cols=[ "age_int",'time_in_hospital','num_lab_procedures', 'num_procedures', 'num_medications',
                      'number_diagnoses',"number_emergency","number_inpatient","number_outpatient",
                        "med_abs_change","med_up_change","med_down_change","healthcare_service_total_visits",
                        ]
            bool_cols=["change","diabetesMed","medicare","medicaid","had_emergency","had_inpatient_days","had_outpatient_days"]
            cols_to_keep = ["race","gender","age_int","discharge_disposition_id","admission_source_id","time_in_hospital",
                            "number_emergency","number_inpatient","number_outpatient",
                                "num_lab_procedures","num_procedures","num_medications","diag_1","diag_2","diag_3","number_diagnoses",
                                "medical_specialty",
                                "max_glu_serum_","A1Cresult_",
                                #"max_glu_serum__","A1Cresult__",
                        #"max_glu_serum","A1Cresult",
                        "insulin","change",
                        "med_abs_change","med_up_change","med_down_change","healthcare_service_total_visits","time_hos_bin",
            "diabetesMed", 
            #"medicare", "medicaid",
            "medicare/medicaid",
             # "had_emergency", "had_inpatient_days", "had_outpatient_days","age"
             #"med_biguanides_and_related_abs_change","med_biguanides_and_related_up_change","med_biguanides_and_related_down_change","med_sulfonylureas_and_meglitinides_abs_change","med_sulfonylureas_and_meglitinides_up_change","med_sulfonylureas_and_meglitinides_down_change","med_thiazolidinediones_and_miscellaneous_abs_change","med_thiazolidinediones_and_miscellaneous_up_change","med_thiazolidinediones_and_miscellaneous_down_change",
            ]




        #Binarize and bin features
        df.loc[:, "medicare"] = (df.payer_code == "MC")
        df.loc[:, "medicaid"] = (df.payer_code == "MD")
        df.loc[:, "medicare/medicaid"] = df["medicare"] + df["medicaid"]

        df.loc[:, "had_emergency"] = (df["number_emergency"] > 0)
        df.loc[:, "had_inpatient_days"] = (df["number_inpatient"] > 0)
        df.loc[:, "had_outpatient_days"] = (df["number_outpatient"] > 0)
        df["change"]=df["change"]!="No"
        df["diabetesMed"]=df["diabetesMed"]!="No"
        #df["readmitted"]=df["readmitted"]=="<30"
        df["max_glu_serum"]=[None if i=="nan" else i for i in df["max_glu_serum"]]
        df["A1Cresult"]=[None if i=="nan" else i for i in df["A1Cresult"]]
        df["max_glu_serum_"]=df["max_glu_serum"].replace({"None":None,"Norm":"Norm",">200":">200",">300":">200"})
        df["A1Cresult_"]=df["A1Cresult"].replace({"None":None,"Norm":"Norm",">7":">7",">8":">7"})
        df["max_glu_serum__"]=df["max_glu_serum"].replace({None:0,"Norm":0,">200":1,">300":1})
        df["A1Cresult__"]=df["A1Cresult"].replace({None:0,"Norm":0,">7":1,">8":1})
        #df[num_cols]=df[num_cols].astype(int)
        
        
        if sqrt == True:
            print("Sqrt transformation...")
            df[num_cols]=np.sqrt(df[num_cols])
            #from sklearn.preprocessing import MinMaxScaler
            #df[num_cols]=RobustScaler(quantile_range=(0,99)).fit_transform(df[num_cols],)
            #scaler=MinMaxScaler()
            #df[num_cols]=scaler.fit_transform(df[num_cols],)

        df[num_cols]=df[num_cols].astype(float)
        df[bool_cols]=df[bool_cols].astype(bool)



        cat_cols = df.select_dtypes(include=["object"]).columns
        df[cat_cols]=df[cat_cols].astype("category")
        
        if return_y==True:
            df["readmitted"]=df["readmitted"]=="<30"
            final_df = df.loc[:, cols_to_keep]
            return final_df,df["readmitted"]
        else:
            final_df = df.loc[:, cols_to_keep]
            return final_df    

from sklearn.base import BaseEstimator, TransformerMixin

class InitialPP_class(BaseEstimator, TransformerMixin):
    def __init__(self,): # no *args or **kargs
        self.scale=None
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, feat_eng=True,sqrt=True,):
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        from custom_classes_and_functions import InitialPP_func
        return InitialPP_func(X,feat_eng=feat_eng,sqrt=sqrt)
    
def CompletePipeline(df,ohe=False):
    """Input: list of categorical columns and numerical columns
            *for LGBM can enter empty list for categorical
       Output: full pipeline with column transformer for preprocessing data"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler, OneHotEncoder,MinMaxScaler
    from sklearn.compose import ColumnTransformer
    from custom_classes_and_functions import InitialPP_class

    #num_pipeline = Pipeline([('mm_scaler', RobustScaler(quantile_range=(0,99))), ])
    initialPP = Pipeline([("initialPP", InitialPP_class(),)])
    num_pipeline = Pipeline([('scaler', MinMaxScaler()), ])
    cat_pipeline = Pipeline([("ohe", OneHotEncoder(),)])

    if ohe==False:

        full_pipeline = ColumnTransformer([ ("num", num_pipeline,df.select_dtypes(include=["float64","int64"]).columns),],remainder="passthrough")
        return full_pipeline
    else:

        full_pipeline = ColumnTransformer([("num", num_pipeline,df.select_dtypes(include=["float64","int64"]).columns),("cat", cat_pipeline,df.select_dtypes(include=["category"]).columns),],remainder="passthrough")
        return full_pipeline

def make_df_from_pipeline_result(pipeline_res,df):
    """Input: pipeline result and original dataframe
       Output: dataframe with transformed data"""
    import pandas as pd
    import numpy as np
    num_cols=df.select_dtypes(include=["float64","int64"]).columns
    cat_cols=df.select_dtypes(include=["category"]).columns
    bool_cols=df.select_dtypes(include=["bool"]).columns
    all_cols=list(num_cols)+list(bool_cols)+list(cat_cols)
    X=pd.DataFrame(pipeline_res,columns=all_cols)
    X[num_cols]=X[num_cols].astype(float)
    X[bool_cols]=X[bool_cols].astype(bool)
    X[cat_cols]=X[cat_cols].astype("category")
    return X



def NumCatPipeline(cat_cols,num_cols):
    """Input: list of categorical columns and numerical columns
            *for LGBM can enter empty list for categorical
       Output: full pipeline with column transformer for preprocessing data"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler, OneHotEncoder,MinMaxScaler
    from sklearn.compose import ColumnTransformer

    #num_pipeline = Pipeline([('mm_scaler', RobustScaler(quantile_range=(0,99))), ])
    num_pipeline = Pipeline([('mm_scaler', MinMaxScaler()), ])

    cat_pipeline = Pipeline([("ohe", OneHotEncoder(),)])


    full_pipeline = ColumnTransformer([("num", num_pipeline,num_cols),("cat", cat_pipeline,cat_cols),],remainder="passthrough")
    return full_pipeline#,full_pipeline.get_feature_names_out(list(num_cols)+list(cat_cols))

def complete_pp(df,ohe=False,feat_eng=True,sqrt=True,):
    import custom_classes_and_functions as ccf
    import pandas as pd

    X_train, X_test, y_train, y_test=ccf.filtering_and_train_test_split(df)
    X_train=ccf.InitialPP_func(X_train,feat_eng=feat_eng,sqrt=sqrt)
    X_test=ccf.InitialPP_func(X_test,feat_eng=feat_eng,sqrt=sqrt)
    num_cols=X_train.select_dtypes(include=["float64","int64"]).columns
    cat_cols=X_train.select_dtypes(include=["category"]).columns
    bool_cols=X_train.select_dtypes(include=["bool"]).columns
    all_cols=list(num_cols)+list(bool_cols)+list(cat_cols)
    X_train=X_train[all_cols].copy()
    X_test=X_test[all_cols].copy()


    if ohe==False:
        nc=ccf.NumCatPipeline(cat_cols=[],num_cols=num_cols)
        X_train=nc.fit_transform(X_train)
        X_test=nc.transform(X_test)
        X_train.shape,X_test.shape
        X_train=pd.DataFrame(X_train,columns=all_cols)
        X_test=pd.DataFrame(X_test,columns=all_cols)
        X_train[num_cols]=X_train[num_cols].astype(float)
        X_test[num_cols]=X_test[num_cols].astype(float)
        X_train[bool_cols]=X_train[bool_cols].astype(bool)
        X_test[bool_cols]=X_test[bool_cols].astype(bool)
        X_train[cat_cols]=X_train[cat_cols].astype("category")
        X_test[cat_cols]=X_test[cat_cols].astype("category")
        return X_train, X_test, y_train, y_test
    
    else:
        nc=ccf.NumCatPipeline(cat_cols=cat_cols,num_cols=num_cols)
        X_train=nc.fit_transform(X_train)
        X_test=nc.transform(X_test)   
        feature_names = nc.get_feature_names_out(all_cols)

        return X_train, X_test, y_train, y_test ,feature_names