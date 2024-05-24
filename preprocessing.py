import pandas as pd

df = pd.read_csv('datasets/compas-scores-raw.csv')

# Drop attributes.
to_drop = ['IsCompleted', 'IsDeleted', 'AssessmentType', 'MiddleName',
           'Person_ID', 'AssessmentID', 'Case_ID', 'Agency_Text',
           'LastName', 'FirstName', 'DateOfBirth', 'Language',
           'RecSupervisionLevel', 'RecSupervisionLevelText', 'DisplayText',
           'RawScore', 'DecileScore', 'Screening_Date', 'ScaleSet']
df = df.drop(to_drop, axis=1)

# Attributes.
Sex_Code_Text = {"Male": 1, "Female": 2}
df["Sex_Code_Text"] = df["Sex_Code_Text"].map(Sex_Code_Text)

Ethnic_Code_Text = {"Caucasian": 1, "African-American": 2, "African-Am": 2}
df["Ethnic_Code_Text"] = df["Ethnic_Code_Text"].map(Ethnic_Code_Text)

LegalStatus = {'Pretrial': 1, 'Post Sentence': 2, 'Conditional Release': 3,
               'Probation Violator': 4, 'Parole Violator': 5, 'Deferred Sentencing': 6, 'Other': 7}
df["LegalStatus"] = df["LegalStatus"].map(LegalStatus)

AssessmentReason = {"Intake": 1}
df["AssessmentReason"] = df["AssessmentReason"].map(AssessmentReason)

CustodyStatus = {'Jail Inmate': 1, 'Probation': 2, 'Pretrial Defendant': 3, 'Residential Program': 4,
                 'Prison Inmate': 5, 'Parole': 6}
df["CustodyStatus"] = df["CustodyStatus"].map(CustodyStatus)

MaritalStatus = {'Single': 1, 'Married': 2, 'Significant Other': 3, 'Divorced': 4, 'Separated': 5, 'Widowed': 6,
                 'Unknown': 7}
df["MaritalStatus"] = df["MaritalStatus"].map(MaritalStatus)

ScaleSet_ID = {22: 1, 17: 2}
df["ScaleSet_ID"] = df["ScaleSet_ID"].map(ScaleSet_ID)

Scale_ID = {7: 1, 8: 2, 18: 3}
df["Scale_ID"] = df["Scale_ID"].map(Scale_ID)

# Output.
ScoreText={"Low":1, "Medium":2, "High":3 }
df["ScoreText"]=df["ScoreText"].map(ScoreText)