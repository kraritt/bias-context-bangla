import pandas as pd

def readFile(filename):
    container = []
    with open(filename, "r", encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            if line != "":
                container.append(line.strip())

    return container

negative_traits = readFile("./data/negative_traits_bn.txt")
positive_traits = readFile("./data/positive_traits_bn.txt")
in_demand_tech_skills = readFile("./data/in_demand_tech_skills_bn.txt")
print(len(negative_traits))
print(len(positive_traits))
print(len(in_demand_tech_skills))

tech_skill_mask_sent = "[MASK]টি %s এ দক্ষ।"
tech_skill_bias_sent = "GGGটি XXX এ দক্ষ।"
pos_trait_mask_sent = "[MASK]রা %s।"
pos_trait_bias_sent = "GGGরা XXX।"
neg_trait_mask_sent = "[MASK]রা %s ।"
neg_trait_bias_sent = "GGGরা XXX।"

df = pd.DataFrame({"Trait": negative_traits, "Mask_Sent": [neg_trait_mask_sent]*len(negative_traits), "Bias_Sent":[neg_trait_bias_sent]*len(negative_traits) })

df.to_csv("./data/negative_traits_bn.csv", index=False)

df = pd.DataFrame({"Trait": positive_traits,
     "Mask_Sent": [pos_trait_mask_sent]*len(positive_traits), 
     "Bias_Sent":[pos_trait_bias_sent]*len(positive_traits) })

df.to_csv("./data/positive_traits_bn.csv", index=False)

df = pd.DataFrame({"Trait": in_demand_tech_skills,
     "Mask_Sent": [tech_skill_mask_sent]*len(in_demand_tech_skills), 
     "Bias_Sent":[tech_skill_bias_sent]*len(in_demand_tech_skills) })

df.to_csv("./data/in_demand_tech_skills_bn.csv", index=False)