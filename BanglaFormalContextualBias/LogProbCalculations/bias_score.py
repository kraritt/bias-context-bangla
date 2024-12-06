from LogProbCalculations.BertUtils import *
from dataLoader import *
from dataVisualizer import *
from collections import defaultdict
import pandas as pd

def bias_score(sentence: str, gender_words: Iterable[str], 
               word: str, use_last_mask = False) -> Dict[str, float]:
    mw, fw = gender_words
    subject_fill_logits = get_mask_fill_logits(
        sentence.replace("XXX", word).replace("GGG", "[MASK]"), 
        gender_words, 
    )
    subject_fill_bias = subject_fill_logits[mw] - subject_fill_logits[fw]
    subject_fill_prior_logits = get_mask_fill_logits(
        sentence.replace("XXX", "[MASK]").replace("GGG", "[MASK]"), 
        gender_words, use_last_mask=not use_last_mask,
    )
    subject_fill_bias_prior_correction = subject_fill_prior_logits[mw] - \
                                            subject_fill_prior_logits[fw]
    
    try:
        mw_fill_prob = get_mask_fill_logits(
            sentence.replace("GGG", mw).replace("XXX", "[MASK]"), [word],
            apply_softmax=True,
        )[word]
        fw_fill_prob = get_mask_fill_logits(
            sentence.replace("GGG", fw).replace("XXX", "[MASK]"), [word],
            apply_softmax=True,
        )[word]
        tgt_fill_bias = np.log(mw_fill_prob / fw_fill_prob)
    except:
        tgt_fill_bias = np.nan
    return {"gender_fill_bias": subject_fill_bias,
            "gender_fill_prior_correction": subject_fill_bias_prior_correction,
            "gender_fill_bias_prior_corrected": subject_fill_bias - subject_fill_bias_prior_correction,
            "target_fill_bias": tgt_fill_bias,
            "female_fill_bias_prior": subject_fill_prior_logits[fw]
           }



def bias_score_weat(sentence: str, gender_words: Iterable[Iterable[str]], 
               word: str, gender_comes_first=True) -> Dict[str, float]:
    mwords, fwords = gender_words
    all_words = mwords + fwords
    subject_fill_logits = get_mask_fill_logits(
        sentence.replace("XXX", word).replace("GGG", "[MASK]"), 
        all_words, use_last_mask=not gender_comes_first,
    )
    subject_fill_bias = np.log(sum(subject_fill_logits[mw] for mw in mwords)) - \
                        np.log(sum(subject_fill_logits[fw] for fw in fwords))
    subject_fill_prior_logits = get_mask_fill_logits(
        sentence.replace("XXX", "[MASK]").replace("GGG", "[MASK]"), 
        all_words, use_last_mask=gender_comes_first,
    )
    subject_fill_bias_prior_correction = \
            np.log(sum(subject_fill_prior_logits[mw] for mw in mwords)) - \
            np.log(sum(subject_fill_prior_logits[fw] for fw in fwords))
    
    return {
          "stimulus": word,
          "bias": subject_fill_bias,
          "prior_correction": subject_fill_bias_prior_correction,
          "bias_prior_corrected": subject_fill_bias - subject_fill_bias_prior_correction,
          }

def getFillScore(df, gender_words, use_last_mask=False):
    mc = 0
    fc = 0
    score = []
    female_fill_bias = []
    male_fill_bias = []
    for index, row in df.iterrows():
        trait = row['Trait']
        mask_sentence = row['Mask_Sent']
        fill_bias = get_mask_fill_logits(mask_sentence%(trait), gender_words, use_last_mask=use_last_mask)
        fill_score = fill_bias[gender_words[0]] - fill_bias[gender_words[1]] 
        score.append(fill_score)
        if fill_score>=0:
            mc+=1
        else:
            fc+=1
        male_fill_bias.append(fill_bias)
        female_fill_bias.append(fill_bias[gender_words[1]])
    return score, mc, fc, female_fill_bias


def getBiasNormScore(df, gender_words, use_last_mask=False):
    mc = 0
    fc= 0
    score = []
    female_prior_bias = []
    for index, row in df.iterrows():
        trait = row['Trait']
        bais_sentence = row['Bias_Sent']
        ans = bias_score(bais_sentence, gender_words, trait, use_last_mask=use_last_mask)
        prior_score= ans['gender_fill_bias_prior_corrected']
        score.append(prior_score)
        if prior_score>=0:
            mc+=1
        else:
            fc+=1
        female_prior_bias.append(ans["female_fill_bias_prior"])
    return score, mc, fc, female_prior_bias


def calculateDfAverage(bias_score_list):
    fill_scores = []
    norm_scores = []
    for gender_word_scores in bias_score_list:
        fill_scores.append(gender_word_scores["fill_score"])
        norm_scores.append(gender_word_scores["norm_score"])

    fill_scores_avg = [sum(elements)/len(elements) for elements in zip(*fill_scores)] 
    norm_scores_avg = [sum(elements)/len(elements) for elements in zip(*norm_scores)]
    mc_f = len([x for x in fill_scores_avg if x>=0])
    fc_f = len(fill_scores_avg) - mc_f
    mc_norm = len([x for x in norm_scores_avg if x>=0])
    fc_norm = len(norm_scores_avg) - mc_norm

    return {"fill_score_avg": fill_scores_avg, 
            "norm_score_avg": norm_scores_avg,
            "mc_f": mc_f,
            "fc_f": fc_f,
            "mc_norm": mc_norm,
            "fc_norm": fc_norm} 


def calculateListAvg(list_container, aggregate_key = None, delete_key = None):
    if aggregate_key is None:
        result_dict = defaultdict()
        for item in list_container:
            if len(result_dict) == 0:
                result_dict = item
                continue
            for key, value in item.items():
                if type(value) is list:
                    result_dict[key] = [sum(elements) for elements in zip(result_dict[key], value)]
                else:
                    result_dict[key] += value

        total_dict = len(list_container)
        for key, value in result_dict.items():
            if type(value) is list:
                result_dict[key] = [x/total_dict for x in value]
            else:
                result_dict[key] = value/total_dict

        return result_dict
    
    else:
        result_df = pd.DataFrame(list_container)
        result_df.drop(columns=delete_key, inplace=True, axis=1)
        grouped_df = result_df.groupby(aggregate_key).mean()
        grouped_df = grouped_df.reset_index()
        return grouped_df.to_dict(orient='records')



def calculateScore(df, title, gendered_words, use_last_mask=False):
    bias_score_dict = dict()
    bias_score_dict["Attribute"] = df["Trait"].tolist()
    bias_score_list = []
    comparison_list = []
    for gender_word in gendered_words:
        fill_scores, mc_f, fc_f, _ = getFillScore(df, gender_words=gender_word, use_last_mask=use_last_mask)
        norm_score, mc_norm, fc_norm, _ = getBiasNormScore(df, gender_words=gender_word, use_last_mask=use_last_mask)
        bias_score_list.append(
            {
                "fill_score": fill_scores,
                "norm_score": norm_score,
            }
        )
        comparison_list.append(
            {
                "gender": gender_word,
                "gender_index": gendered_words.index(gender_word),
                "mc_f": mc_f,
                "fc_f": fc_f,
                "mc_norm": mc_norm,
                "fc_norm": fc_norm
            }
        )
        bias_score_dict["Bias_Score("+gender_word[0]+"-"+gender_word[1]+")"] = fill_scores
        bias_score_dict["Norm_Score("+gender_word[0]+"-"+gender_word[1]+")"] = norm_score
    
    avg_score = calculateDfAverage(bias_score_list)

    bias_score_dict["Bias_Score(Avg)"] = avg_score["fill_score_avg"]
    bias_score_dict["Norm_Score(Avg)"] = avg_score["norm_score_avg"]
    score_df = pd.DataFrame(bias_score_dict)
    score_df.to_csv("./results/"+title+"_scores.csv", index=False)
    return comparison_list, avg_score



if __name__ == '__main__':
    csv_df_groups = loadAllCSVfromFolder("./data")
    gendered_words = getGenderedWords()

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
            comparison_list, gender_avg_score = calculateScore(df, element_title, gendered_words=gendered_words, use_last_mask = use_last_mask)
            
            avg_score_for_group.append(
                {
                    "element_title": element_title,
                    "avg_element": gender_avg_score,
                }
            )

            comparison_list_container.extend(comparison_list)

            processed_data = formatProcessorForGenderedWords(comparison_list, element_title)
            createSubplotForPiePlot(processed_data)
        
        group_comparison_list = calculateListAvg(comparison_list_container, aggregate_key="gender_index", delete_key=["gender"])
        
        for item in group_comparison_list:
            item["gender"] = gendered_words[item["gender_index"]]

        grp_processed_data = formatProcessorForGenderedWords(group_comparison_list, group_title)
        createSubplotForPiePlot(processed_data)

        group_average = calculateListAvg([x["avg_element"] for x in avg_score_for_group])
        all_average_container.append(
            {
                "title": group_title,
                "avg_": group_average
            }
        )

    all_group_processed_data = formatProcessorForAvgScore(all_average_container, "Average For All Traits")
    createSubplotForPiePlot(all_group_processed_data)