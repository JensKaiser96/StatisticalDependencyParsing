from src.tools.io import read_file
from src.tools.measurements import UAS, LAS


def main():
    gold_tree_bank = read_file("data/english/dev/wsj_dev.conll06.gold")
    pred_tree_bank = read_file("data/english/dev/wsj_dev.conll06.pred")

    print(f"UAS: {UAS.__call__(gold_tree_bank, pred_tree_bank)}")
    print(f"LAS: {LAS.__call__(gold_tree_bank, pred_tree_bank)}")


if __name__ == '__main__':
    main()

