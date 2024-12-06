import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import math

font_prop = fm.FontProperties(fname='./kalpurush.ttf')

def createSubplotForPiePlot(data, save=True):
    title = data["title"]
    data = data["data"]
    if math.ceil((len(data)*2)/4) > 1:
      fig, axs = plt.subplots(math.ceil((len(data)*2)/4), 4, figsize=(20, 10))
      cols = 4
    else:
      fig, axs = plt.subplots(2, 2, figsize=(20, 10))
      cols = 2
    fig.suptitle(title)
    for i, d in enumerate(data):
        labels = d["labels"]
        values = d["values"]
        norm_values = d["norm_values"]
        _, texts, _ = axs[(2*i)//cols,(2*i)%cols].pie(values, labels=labels, autopct='%.2f%%')
        axs[(2*i)//cols,(2*i)%cols].set_title(d["subTitle"]+"_Fill Score", fontproperties=font_prop)
        for text in texts:
          text.set_fontproperties(font_prop)
        
        _, texts, _ = axs[(2*i + 1)//cols,(2*i + 1)%cols].pie(norm_values, labels=labels, autopct='%.2f%%')
        axs[(2*i + 1)//cols,(2*i + 1)%cols].set_title(d["subTitle"]+"_Norm Score", fontproperties=font_prop)
        
        for text in texts:
          text.set_fontproperties(font_prop)
    if save:
        plt.savefig("./results/"+title+".png")
    plt.show()


def formatProcessorForAvgScore(avg_score_list, title):
    container = {}
    container["title"] = title
    container["data"] = []
    for i, avg_score in enumerate(avg_score_list):
        d = {
            "subTitle": "_".join(avg_score["title"].split("_")[:-1]),
            "labels": ["Male Word", "Female Word"],
            "values": [avg_score["avg_"]["mc_f"], avg_score["avg_"]["fc_f"]],
            "norm_values": [avg_score["avg_"]["mc_norm"], avg_score["avg_"]["fc_norm"]]
        }
        container["data"].append(d)
    return container

def formatProcessorForGenderedWords(compare_list, title):
    container = {}
    container["title"] = title
    container["data"] = []
    for i, c in enumerate(compare_list):
        d = {
            "subTitle": u"%s"%(c["gender"][0]) + "-" + u"%s"%(c["gender"][1]),
            "labels": [u"%s"%(gender_word) for gender_word in c["gender"]],
            "values": [c["mc_f"], c["fc_f"]],
            "norm_values": [c["mc_norm"], c["fc_norm"]]
        }
        container["data"].append(d)
    return container