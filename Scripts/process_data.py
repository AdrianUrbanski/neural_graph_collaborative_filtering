import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Amazon data.")
    parser.add_argument(
        'path',
        type=str,
        help='path in which to store the results',
    )
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='only process the given parquet file; if empty processes every file',
    )
    parser.add_argument(
        '--data_path',
        dest='data_path',
        type=str,
        default='/pio/scratch/1/recommender_systems/interim/Amazon',
        help='path in which data files are',
    )

    return parser.parse_args()


def remap_pdf(pdf, save_path):
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    pdf['reviewerID'] = user_encoder.fit_transform(pdf['reviewerID'])
    pdf['asin'] = item_encoder.fit_transform(pdf['asin'])

    user_list_pdf = pd.DataFrame(zip(user_encoder.classes_, range(len(user_encoder.classes_))))
    user_list_pdf.columns = ['org_id', 'remap_id']
    user_list_pdf.to_csv(save_path / 'user_list.txt', sep=' ', index=False, header=True)

    item_list_pdf = pd.DataFrame(zip(item_encoder.classes_, range(len(item_encoder.classes_))))
    item_list_pdf.columns = ['org_id', 'remap_id']
    item_list_pdf.to_csv(save_path / 'item_list.txt', sep=' ', index=False, header=True)

    return pdf


def main():
    args = parse_args()
    tqdm.pandas()

    data_root = Path(args.data_path)
    save_path = Path(args.path)

    if args.file is None:
        pattern = re.compile("^(?!meta).*clean\.parquet$")
        data = []
        for file in data_root.iterdir():
            if pattern.match(file.name):
                data.append(pd.read_parquet(file))
        data_pdf = pd.concat(data)
    else:
        data_pdf = pd.read_parquet(data_root / args.file)

    data_pdf = remap_pdf(data_pdf, save_path)

    train_pdf = data_pdf.sample(frac=0.9)
    test_pdf = data_pdf[~data_pdf.index.isin(train_pdf.index)]
    test_pdf = test_pdf[test_pdf['reviewerID'].isin(train_pdf['reviewerID']) & test_pdf['asin'].isin(train_pdf['asin'])]

    grouped_train_pdf = train_pdf.groupby('reviewerID').progress_apply(
        lambda pdf: f"{pdf.iloc[0]['reviewerID']} " + ' '.join(map(str, pdf['asin']))
    )

    grouped_test_pdf = test_pdf.groupby('reviewerID').progress_apply(
        lambda pdf: f"{pdf.iloc[0]['reviewerID']} " + ' '.join(map(str, pdf['asin']))
    )

    grouped_train_pdf.to_csv(save_path / 'train.txt', header=False, index=False)
    grouped_test_pdf.to_csv(save_path / 'test.txt', header=False, index=False)


if __name__ == '__main__':
    main()
