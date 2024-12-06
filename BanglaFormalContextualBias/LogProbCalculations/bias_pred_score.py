from LogProbCalculations.BertUtils import *
from dataLoader import *
from dataVisualizer import *
from collections import defaultdict
import pandas as pd
import numpy as np


def convertToDictOfLists(container):
    dict_of_lists = defaultdict(list)
    for d in container:
        for k, v in d.items():
            if k in dict_of_lists.keys():
                dict_of_lists[k].append(v)
            else:
                dict_of_lists[k] = [v]
    return dict_of_lists


def getNormalizedPredsValue(male_preds, female_preds):
    normalizedMalePreds = (male_preds) / (male_preds + female_preds)
    normalizedFemalePreds = (female_preds) / (male_preds + female_preds)

    return normalizedMalePreds.tolist(), normalizedFemalePreds.tolist()


def bias_score_aux(
    sentence,
    trait,
    gender_words,
    use_last_mask,
    apply_softmax=False,
    calc_tgt_fill_prob=False,
):
    subject_fill_logits = get_mask_fill_logits(
        sentence.replace("XXX", trait).replace("GGG", "[MASK]"),
        gender_words,
        apply_softmax=apply_softmax,
    )
    subject_fill_prior_logits = get_mask_fill_logits(
        sentence.replace("XXX", "[MASK]").replace("GGG", "[MASK]"),
        gender_words,
        use_last_mask=not use_last_mask,
        apply_softmax=apply_softmax,
    )

    norm_logits = {
        gw: subject_fill_logits[gw] - subject_fill_prior_logits[gw]
        for gw in gender_words
    }

    tgt_fill_prob = {}

    if calc_tgt_fill_prob:
        for gw in gender_words:
            tgt_fill_prob[gw] = get_mask_fill_logits(
                sentence.replace("GGG", gw).replace("XXX", "[MASK]"),
                [trait],
                apply_softmax=True,
            )[trait]

    return norm_logits, tgt_fill_prob


def getFillScore(df, gender_words, use_last_mask):
    fill_logits_container = []
    fill_preds_container = []
    for index, row in df.iterrows():
        trait = row["Trait"]
        mask_sentence = row["Mask_Sent"]
        fill_logits = get_mask_fill_logits(
            mask_sentence % (trait), gender_words, use_last_mask=use_last_mask
        )
        fill_preds = get_mask_fill_logits(
            mask_sentence % (trait),
            gender_words,
            apply_softmax=True,
            use_last_mask=use_last_mask,
        )

        fill_logits_container.append(fill_logits)
        fill_preds_container.append(fill_preds)

    return convertToDictOfLists(fill_logits_container), convertToDictOfLists(
        fill_preds_container
    )


def getBiasNormScore(df, gender_words, use_last_mask):
    norm_logits_container = []
    norm_preds_container = []
    target_fill_prob_container = []
    for index, row in df.iterrows():
        trait = row["Trait"]
        bias_sentence = row["Bias_Sent"]

        norm_logits, _ = bias_score_aux(
            bias_sentence, trait, gender_words, use_last_mask=use_last_mask
        )
        norm_preds, target_fill_prob = bias_score_aux(
            bias_sentence,
            trait,
            gender_words,
            use_last_mask,
            apply_softmax=True,
            calc_tgt_fill_prob=True,
        )
        norm_logits_container.append(norm_logits)
        norm_preds_container.append(norm_preds)
        target_fill_prob_container.append(target_fill_prob)
    return (
        convertToDictOfLists(norm_logits_container),
        convertToDictOfLists(norm_preds_container),
        convertToDictOfLists(target_fill_prob_container),
    )


def calculateLogitScores(df, title, gendered_words, use_last_mask=False):
    bias_score_dict = dict()
    bias_score_dict["Attribute"] = df["Trait"].tolist()

    gender_fill_logits, gender_fill_preds = getFillScore(
        df, gendered_words, use_last_mask
    )
    norm_logits, norm_preds, target_fill_prob = getBiasNormScore(
        df, gendered_words, use_last_mask
    )

    aggregate_male_preds = np.zeros(len(bias_score_dict["Attribute"]))
    aggregate_female_preds = np.zeros(len(bias_score_dict["Attribute"]))

    mean_norm_score = np.zeros((len(bias_score_dict["Attribute"]), 1), dtype=np.float32)
    mean_norm_score_logits = np.zeros(
        (len(bias_score_dict["Attribute"]), 1), dtype=np.float32
    )

    for i in range(0, len(gendered_words), 2):
        mw = gendered_words[i]
        fw = gendered_words[i + 1]
        bias_score_dict["logits-" + mw] = gender_fill_logits[mw]
        bias_score_dict["logits-" + fw] = gender_fill_logits[fw]

        bias_score_dict["normlogits-" + mw] = norm_logits[mw]
        bias_score_dict["normlogits-" + fw] = norm_logits[fw]

        bias_score_dict["preds-" + mw] = gender_fill_preds[mw]
        bias_score_dict["preds-" + fw] = gender_fill_preds[fw]

        bias_score_dict["preds_diff(" + mw + "-" + fw + ")"] = [
            x - y for x, y in zip(gender_fill_preds[mw], gender_fill_preds[fw])
        ]
        bias_score_dict["normlogits_diff(" + mw + "-" + fw + ")"] = [
            x - y for x, y in zip(norm_logits[mw], norm_logits[fw])
        ]
        mean_norm_score_logits = np.column_stack(
            (
                mean_norm_score_logits,
                bias_score_dict["normlogits_diff(" + mw + "-" + fw + ")"],
            )
        )

        aggregate_male_preds = aggregate_male_preds + gender_fill_preds[mw]
        aggregate_female_preds = aggregate_female_preds + gender_fill_preds[fw]

        bias_score_dict["normpreds-" + mw] = norm_preds[mw]
        bias_score_dict["normpreds-" + fw] = norm_preds[fw]

        bias_score_dict["normpreds_diff(" + mw + "-" + fw + ")"] = [
            x - y for x, y in zip(norm_preds[mw], norm_preds[fw])
        ]
        mean_norm_score = np.column_stack(
            (mean_norm_score, bias_score_dict["normpreds_diff(" + mw + "-" + fw + ")"])
        )

        bias_score_dict["tgtfillprob-" + mw] = target_fill_prob[mw]
        bias_score_dict["tgtfillprob-" + fw] = target_fill_prob[fw]

    (
        bias_score_dict["Bias_Score(Male_Aggregate)"],
        bias_score_dict["Bias_Score(Female_Aggregate)"],
    ) = getNormalizedPredsValue(aggregate_male_preds, aggregate_female_preds)

    mean_norm_score_logits = np.delete(mean_norm_score_logits, 0, axis=1)
    mean_norm_score_logits = np.mean(mean_norm_score_logits, axis=1)
    bias_score_dict["Mean_Norm_Score(Logits)"] = mean_norm_score_logits.tolist()

    mean_norm_score = np.delete(mean_norm_score, 0, axis=1)
    mean_norm_score = np.mean(mean_norm_score, axis=1)

    bias_score_dict["Mean_Norm_Score(Preds)"] = mean_norm_score.tolist()

    score_df = pd.DataFrame(bias_score_dict)
    score_df.to_csv("./results/" + title + "_scores.csv", index=False)


if __name__ == "__main__":
    csv_df_groups = loadAllCSVfromFolder("./data")
    gendered_words_list = getGenderedWords()
    gendered_words = []

    for lst in gendered_words_list:
        gendered_words.extend(lst)

    all_average_container = []
    for group in csv_df_groups:
        group_title = group["title"]
        elements = group["group"]

        avg_score_for_group = []
        comparison_list_container = []

        for element in elements:
            element_title = element["title"]
            df = element["df"]
            use_last_mask = element["use_last_mask"]
            print("Processing: ", element_title)
            calculateLogitScores(
                df,
                element_title,
                gendered_words=gendered_words,
                use_last_mask=use_last_mask,
            )
