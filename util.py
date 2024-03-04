#* utilize transformers [type of nn]
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# pytorch
import torch
from typing import Tuple 
# determine if gpu or cpu is being ussed 
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#* finbert model for estimating sentiment... attribute values to words/headlines
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
#* assign the token of model into a model
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
#* the binary classifications 
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news):
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        # sum the result of the headline numbers and take the highest result
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        # convert to probability 
        probability = result[torch.argmax(result)]
        # label the probability as our sentiment 
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]

#* Test case, variable input if interested 
if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment(['markets responded with a downward trajectory to the news!','traders were not too happy with today!'])
    print(tensor, sentiment)
    print(torch.cuda.is_available())