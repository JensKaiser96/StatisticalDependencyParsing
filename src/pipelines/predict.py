from src.tools.io import read_file, write_file


def run_prediction_pipeline():
    dev_pred_file_path = "./../../data/english/dev/wsj_dev.conll06.blind"
    dev_pred = read_file(dev_pred_file_path)

    for sentence in dev_pred:
        graph = sentence2graph(sentence)
        
