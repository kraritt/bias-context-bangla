import pandas as pd
import os
from stringGenerator import getFormattedStruSentence, processTraitSuffix


def getDfColumnSets(all_columns):
    num_sets = (len(all_columns) - 1) // 3
    columns_set = [["Trait"] for i in range(num_sets)]
    assert len(columns_set) == num_sets

    for column in all_columns:
        if column == "Trait":
            continue
        else:
            idx = int(column[1]) - 1
            columns_set[idx].append(column)
    return columns_set


def csvLoader(filename, defaultPath="./data/"):
    df = pd.read_csv(os.path.join(defaultPath, filename))
    csv_elements_list = []
    column_sets = getDfColumnSets(df.columns)
    for column_set in column_sets:
        df_aux = df[column_set]
        csv_elements_list.append(
            {"df": df_aux, "title": filename.split(".")[0] + "_" + column_set[1][:2]}
        )
    return csv_elements_list


def fillSentence(sentence, row, sent_idx):
    sentence = sentence.format(
        Mask_Suffix=row["S" + str(sent_idx) + "_" + "Mask_Suffix"],
        PlaceHolder=row["S" + str(sent_idx) + "_" + "Placeholder"].strip(),
        Trait_Suffix=processTraitSuffix(
            row["S" + str(sent_idx) + "_" + "Trait_Suffix"]
        ),
    )

    return sentence


def processDFSentence(csv_files_list):
    for csv_file_list in csv_files_list:
        df = csv_file_list["df"]
        df.fillna("", inplace=True)
        columns = df.columns.tolist()
        sent_idx = int(csv_file_list["title"][-1])
        mask_sent = getFormattedStruSentence(sent_idx)
        bias_sent = mask_sent.replace("[MASK]", "GGG").replace("%s", "XXX")
        df["Mask_Sent"] = df.apply(
            lambda row: fillSentence(mask_sent, row, sent_idx),
            axis=1,
        )
        df["Bias_Sent"] = df.apply(
            lambda row: fillSentence(bias_sent, row, sent_idx),
            axis=1,
        )
        columns.remove("Trait")
        df.drop(columns, axis=1, inplace=True)
        csv_file_list["df"] = df
        csv_file_list["use_last_mask"] = False if sent_idx <= 2 else True
    return csv_files_list


def loadAllCSVfromFolder(folderPath="data/"):
    import os

    files = os.listdir(folderPath)
    csvFiles = []
    for file in files:
        if file.endswith(".csv"):
            csvFiles.append(file)

    csv_files_list = []
    for i in range(len(csvFiles)):
        csv_element = csvLoader(csvFiles[i])
        group_title = csvFiles[i].split(".")[0]
        csv_files_list.append(
            {
                "title": group_title,
                "group": processDFSentence(csv_element),
            }
        )

    return csv_files_list


def getGenderedWords():
    return [["ছেলে", "মেয়ে"], ["পুরুষ", "নারী"], ["যুবক", "যুবতী"], ["বালক", "বালিকা"]]


if __name__ == "__main__":
    csv_files_list = loadAllCSVfromFolder()
    for csv_file in csv_files_list:
        print(csv_file["title"])
        elements = csv_file["group"]
        print("Groups")
        for csv_file_list in elements:
            print(csv_file_list["title"])
            print(csv_file_list["use_last_mask"])
            csv_file_list["df"].to_csv(
                "./example/" + csv_file_list["title"] + ".csv", index=False
            )
