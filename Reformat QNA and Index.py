import pandas
import pandas as pd
import os
import sys

for i, path in enumerate(os.listdir("Data")):
    category = path.split("-")[-1].lstrip(" ").lower().replace(" ", "_")
    print(f"{i+1}: \"{category}\",")

sys.exit()

for path in os.listdir("Data"):
    category = path.split("-")[-1].lstrip(" ")

    ind_path = os.path.join("Data", path, "processed", "index.csv")
    qna_path = os.path.join("Data", path, "processed", "qna.csv")

    df_ind = pandas.read_csv(ind_path)

    if "category" in df_ind.columns:
        df_ind.rename(columns={"category": "Category"}, inplace=True)
    if "Description" in df_ind.columns:
        df_ind.rename(columns={"Description": "Content"}, inplace=True)

    print(df_ind.columns)

    df_ind.to_csv(ind_path, index=False)


    df_qna = pd.read_csv(qna_path)

    if "Question" in df_qna.columns:
        df_qna.rename(columns={"Question": "Title"}, inplace=True)
    if "Answer" in df_qna.columns:
        df_qna.rename(columns={"Answer": "Content"}, inplace=True)
    if "Category" not in df_qna.columns:
        df_qna["Category"] = [category] * len(df_qna)
    if "Severity" not in df_qna.columns:
        df_qna["Severity"] = [0] * len(df_qna)

    df_qna.to_csv(qna_path, index=False)
