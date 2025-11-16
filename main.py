import sys
import yaml
from transformers import pipeline

config_file = 'config.yaml'

if __name__ == '__main__':
    # parse config
    print('Parsing config')
    try:
        with open(config_file, 'r') as file:
            models = yaml.safe_load(file)

        for model in models.get('models', []):
            print(model)
    except Exception as e:
        print('Failed to parse config [{config_file}]')
        sys.exit(1)

    print('Allah Hu')
    # model = pipeline('sentiment-analysis', 'distilbert-base-uncased-finetuned-sst-2-english', framework='pt')
