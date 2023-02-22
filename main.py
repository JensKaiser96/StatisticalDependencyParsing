from src.features.template import TemplateWizard
from src.models.perceptron import Perceptron
from src.tools.CONLL06 import TreeBank
from src.tools.measurements import UAS


def main():
    gold_tree_bank_path = "data/english/train/wsj_train.conll06"
    tb = TreeBank.from_file(gold_tree_bank_path)
    feature_dict = TemplateWizard.create_feature_dict(tb, gold_tree_bank_path + ".feature_dict")

    model = Perceptron(feature_dict, logging=False)
    model_weight_path = f"data/models/{gold_tree_bank_path.split('/')[-1]}"
    train = True
    if train:
        model.train(tb, 50, save_path=model_weight_path)
    else:
        model.load_weights(model_weight_path)
    blind_tree_bank_path = "data/english/test/wsj_test.conll06.blind"
    blind_tree_bank = TreeBank.from_file(blind_tree_bank_path)
    model.annotate(blind_tree_bank)

    print(UAS(tb, blind_tree_bank))


if __name__ == '__main__':
    main()
