from abc import ABC, abstractmethod
from normalizer import normalize
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForPreTraining


class ModelWrapper(ABC):
    def setEmbeddingLayer(self, layer: int) -> None:
        self.layer = layer

    @abstractmethod
    def getWordVector(
        self, word: str, sent: str, index: int, span: list[int]
    ) -> np.array:
        pass


class MLMEmbeddingExtractor(ModelWrapper):
    def __init__(self, model_name: str, tokenizer_name: str) -> None:
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        ) 

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        self.layer = 24

    def getTokenIndices(self, offset_mapping, index, span):
        indices = []
        for i, token in enumerate(offset_mapping):
            if i < index or i == 0:
                continue
            if token[0] >= span[0] and token[0] < span[1]:
                indices.append(i)
            elif token[0] > span[1]:
                break
        assert len(indices) > 0
        return indices

    def prepareEmbedding(self, output, tokenIndices, averagePooling=True):
        if averagePooling:
            return np.mean(
                output[1][self.layer][0].detach().cpu().numpy()[tokenIndices], axis=0
            )
        else:
            return np.max(
                output[1][self.layer][0].detach().cpu().numpy()[tokenIndices], axis=0
            )

    def getWordVector(
        self, word: str, sent: str, index: int, span: list[int]
    ) -> np.array:
        normalized_sentence = normalize(sent) 

        input_tokens = self.tokenizer(normalized_sentence, return_tensors="pt")

        input_token_offsets = self.tokenizer(
            normalized_sentence,
            return_offsets_mapping=True,
            return_tensors="pt",
        ).offset_mapping[0]

        token_indices = self.getTokenIndices(input_token_offsets, index, span)
        if torch.cuda.is_available():
            input_tokens = input_tokens.to("cuda")
        with torch.no_grad():
            output = self.model(**input_tokens)

            return self.prepareEmbedding(output, token_indices)


class BanglaBertDiscriminator(ModelWrapper):
    def __init__(self, model_name: str, tokenizer_name: str) -> None:
        self.model = AutoModelForPreTraining.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        )

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        self.layer = 24

    def getTokenIndices(self, offset_mapping, index, span):
        indices = []
        for i, token in enumerate(offset_mapping):
            if i < index or i == 0:
                continue
            if token[0] >= span[0] and token[0] < span[1]:
                indices.append(i)
            elif token[0] > span[1]:
                break
        assert len(indices) > 0
        return indices

    def prepareEmbedding(self, output, tokenIndices, averagePooling=True):
        if averagePooling:
            return np.mean(
                output[1][self.layer][0].detach().cpu().numpy()[tokenIndices], axis=0
            )
        else:
            return np.max(
                output[1][self.layer][0].detach().cpu().numpy()[tokenIndices], axis=0
            )

    def getWordVector(
        self, word: str, sent: str, index: int, span: list[int]
    ) -> np.array:
        normalized_sentence = normalize(sent) 

        input_tokens = self.tokenizer(normalized_sentence, return_tensors="pt")

        input_token_offsets = self.tokenizer(
            normalized_sentence,
            return_offsets_mapping=True,
            return_tensors="pt",
        ).offset_mapping[0]

        token_indices = self.getTokenIndices(input_token_offsets, index, span)
        if torch.cuda.is_available():
            input_tokens = input_tokens.to("cuda")
        with torch.no_grad():
            output = self.model(**input_tokens)

            return self.prepareEmbedding(output, token_indices)


if __name__ == "__main__":
    model = MLMEmbeddingExtractor(
        model_name="csebuetnlp/banglabert_large",
        tokenizer_name="csebuetnlp/banglabert_large",
    )

    sent1 = "১৫০ টাকা নিয়েছিল। গোলাপ গ্রামের মজার একটা ব্যাপার লক্ষ করেছিলাম। সেখানে সব বাড়ির সাথেই লাগোয়া ছোটছোট গোলাপের বাগান আছে। গাড়ি নিয়ে স্বপরিবারে বেড়াতে যাওয়ার প্ল্যান করার আগে অবশ্যই নিরাপত্তার ব্যপারটি মাথায় রাখতে হবে। পরিবারের নিরাপত্তায় সবার সাথে ফোন এবং ফোনে রিচার্জ করে নিলে ভাল হয়।"
    sent2 = "যথাযথ কর্তৃপক্ষের উচিত এই সকল নিদর্শনসমুহের নিয়মিত পরিচর্যা করা, নতুবা এই সকল নিদর্শনসমুহ একসময় কালের গর্ভে বিলীন হয়ে যাবে। গোলাপ রাজ্য (ভ্রমণ কাহিনী) অভিজিৎ সাগর A rose for my rose.......এটার বদলে যদি বলি a kingdom of rose for my beautiful rose ? কেমন হবে বলুন তো?- যা হবে তা ভাবনাতেই থাকুক।"
    sent3 = "নতুবা এই সকল নিদর্শনসমুহ একসময় কালের গর্ভে বিলীন হয়ে যাবে। গোলাপের রাজ্য"
    sent4 = "গোলাপের রাজ্য"
    sent5 = "নতুবা এই সকল নিদর্শনসমুহ একসময় কালের গর্ভে বিলীন হয়ে যাবে। গোলাপের। রাজ্য"

    print(model.getWordVector("গোলাপের", sent1, 0, [0, 0]))
