def GetEmbeddingsFileMapping():
    embeddingsMapper = {
        "BanglaBert_Generator": "embeddings_len_%s.pkl",
        "BanglaBert_Discriminator": "embeddings_bbdisc_len_%s.pkl",
        "Muril_Base": "embeddings_murilB_len_%s.pkl",
        "XLM_Roberta_Base": "embeddings_xlmRB_len_%s.pkl",
        "XLM_Roberta_Large": "embeddings_XLM_Roberta_Large_len_%s.pkl",
    }

    return embeddingsMapper


def GetOperationalSentenceLengths():
    convergenceSentenceLengths = [
        "9",
        "15",
        "25",
        "40",
        "60",
        "75",
        "100",
        "125",
        "150",
        "200",
    ]
    return convergenceSentenceLengths
