import sys
import yaml
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

config_file = 'config.yaml'

if __name__ == '__main__':
    # parse config
    print('Parsing config')
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Failed to parse config [{config_file}]')
        sys.exit(1)

    # load models and tokenizers from HF
    for m in config.get('models', []):
        print('Loading tokenizer and model from HF')
        tokenizer = AutoTokenizer.from_pretrained(m['name'])
        model = ORTModelForSequenceClassification.from_pretrained(m['name'], export=True)
        print(f"Tokenizer and model loaded from HF [{m['name']}]")

        # preparing input (preprocessing)
        text = 'I love Allah'
        inputs = tokenizer(text, return_tensors='np')

        # run inference
        outputs = model(**inputs)
        logits = outputs.logits

        # post process (softmax + class with highest prob)
        logits_exp = np.exp(logits)
        probs = logits_exp / np.sum(logits_exp, axis=1, keepdims=True)
        pred = np.argmax(probs, axis=1)[0]

        labels = model.config.id2label
        pred_label = labels[pred]

        result = {"label": labels[pred], "score": probs[0, pred]}
        print(result)
        # pred_label = labels[pred]
        # print(probs[0, ])
        # print(pred_label)

    # model = pipeline('sentiment-analysis', 'distilbert-base-uncased-finetuned-sst-2-english', framework='pt')
