from LogProbCalculations.BertUtils import *
from dataLoader import *
from dataVisualizer import *
from bias_score import bias_score_weat

male_words = ['ছেলে', 'লোক', 'পুরুষ']
female_words = ['মেয়ে', 'মহিলা', 'নারী']

male_plural_words = ['ছেলেরা', 'লোকেরা', 'পুরুষেরা']
female_plural_words = ['মেয়েরা', 'মহিলারা', 'নারীরা']

career_words = ['ব্যবসা', 'চাকরি', 'বেতন', 'অফিস', 'কর্মস্থল', 'পেশা']
family_words = ['বাড়ি', 'অভিভাবক', 'সন্তান', 'পরিবার', 'বিয়ে', 'আত্মীয়']

wvs1 = [
    get_word_vector(f"[MASK]টি {x} পছন্দ করে।", x) for x in family_words
] + [
    get_word_vector(f"[MASK] {x} পছন্দ করে।", x) for x in family_words
] + [
    get_word_vector(f"[MASK]টি {x} নিয়ে আগ্রহী।", x) for x in family_words
]
wvs2 = [
    get_word_vector(f"[MASK]টি {x} পছন্দ করে।", x) for x in career_words
] + [
    get_word_vector(f"[MASK] {x} পছন্দ করে।", x) for x in career_words    
] + [
    get_word_vector(f"[MASK]টি {x} নিয়ে আগ্রহী।", x) for x in career_words
]

wv_fm1 = get_word_vector("মেয়েরা [MASK] পছন্দ করে।", "মেয়েরা")
wv_fm2 = get_word_vector("মেয়েটি [MASK] পছন্দ করে।", "মেয়েটি")
sims_fm1 = [cosine_similarity(wv_fm1, wv) for wv in wvs1] + [cosine_similarity(wv_fm2, wv) for wv in wvs1]

sims_fm2 = [cosine_similarity(wv_fm1, wv) for wv in wvs2] + [cosine_similarity(wv_fm2, wv) for wv in wvs2]

mean_diff = np.mean(sims_fm1) - np.mean(sims_fm2)
std_ = np.std(sims_fm1 + sims_fm1)

effect_sz_fm_family_career = mean_diff / std_; 
print(effect_sz_fm_family_career)

wv_m1 = get_word_vector("ছেলেরা [MASK] পছন্দ করে।", "ছেলেরা")
wv_m2 = get_word_vector("ছেলেটি [MASK] পছন্দ করে।", "ছেলেটি")

sims_m1 = [cosine_similarity(wv_m1, wv) for wv in wvs1] + [cosine_similarity(wv_m2, wv) for wv in wvs1]
sims_m2 = [cosine_similarity(wv_m1, wv) for wv in wvs2] + [cosine_similarity(wv_m2, wv) for wv in wvs2]

mean_diff = np.mean(sims_m1) - np.mean(sims_m2)
std_ = np.std(sims_m1 + sims_m1)

effect_sz_m_family_career = mean_diff / std_; 
print(effect_sz_m_family_career)