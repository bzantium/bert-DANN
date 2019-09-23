"""Main script for ADDA."""

import param
from train import train, evaluate
from model import BertEncoder, DistilBertEncoder, BertClassifier, \
    RobertaEncoder, RobertaClassifier, DomainClassifier, RobertaDomainClassifier
from utils import XML2Array, CSV2Array, convert_examples_to_features, \
    roberta_convert_examples_to_features, get_data_loader, init_model
from sklearn.model_selection import train_test_split
from pytorch_transformers import BertTokenizer, RobertaTokenizer
import torch
import os
import random
import argparse


def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--src', type=str, default="books",
                        choices=["books", "dvd", "electronics", "kitchen", "blog", "airline"],
                        help="Specify src dataset")

    parser.add_argument('--tgt', type=str, default="dvd",
                        choices=["books", "dvd", "electronics", "kitchen", "blog", "airline"],
                        help="Specify tgt dataset")

    parser.add_argument('--train', default=False, action='store_true',
                        help='Force to train source encoder/classifier')

    parser.add_argument('--seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--train_seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--load', default=False, action='store_true',
                        help="Load saved model")

    parser.add_argument('--model', type=str, default="bert",
                        choices=["bert", "distilbert", "roberta"],
                        help="Specify model type")

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help="Specify maximum sequence length")

    parser.add_argument('--batch_size', type=int, default=64,
                        help="Specify batch size")

    parser.add_argument('--alpha', type=float, default=1e-3,
                        help="Specify domain weight")

    parser.add_argument('--beta', type=float, default=1.0,
                        help="Specify domain weight")

    parser.add_argument('--num_epochs', type=int, default=3,
                        help="Specify the number of epochs for adaptation")

    parser.add_argument('--log_step', type=int, default=1,
                        help="Specify log step size for adaptation")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_arguments()
    # argument setting
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("alpha: " + str(args.alpha))
    print("beta: " + str(args.beta))
    print("seed: " + str(args.seed))
    print("train_seed: " + str(args.train_seed))
    print("model_type: " + str(args.model))
    print("max_seq_length: " + str(args.max_seq_length))
    print("batch_size: " + str(args.batch_size))
    print("num_epochs: " + str(args.num_epochs))
    set_seed(args.train_seed)

    if args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # preprocess data
    print("=== Processing datasets ===")
    if args.src == 'blog':
        src_x, src_y = CSV2Array(os.path.join('data', args.src, 'blog.csv'))

    elif args.src == 'airline':
        src_x, src_y = CSV2Array(os.path.join('data', args.src, 'airline.csv'))

    else:
        src_x, src_y = XML2Array(os.path.join('data', args.src, 'negative.review'),
                               os.path.join('data', args.src, 'positive.review'))

    src_x, src_test_x, src_y, src_test_y = train_test_split(src_x, src_y,
                                                            test_size=0.2,
                                                            stratify=src_y,
                                                            random_state=args.seed)

    if args.tgt == 'blog':
        tgt_x, tgt_y = CSV2Array(os.path.join('data', args.tgt, 'blog.csv'))

    elif args.tgt == 'airline':
        tgt_x, tgt_y = CSV2Array(os.path.join('data', args.tgt, 'airline.csv'))
    else:
        tgt_x, tgt_y = XML2Array(os.path.join('data', args.tgt, 'negative.review'),
                                 os.path.join('data', args.tgt, 'positive.review'))

    tgt_train_x, _, tgt_train_y, _ = train_test_split(tgt_x, tgt_y,
                                                      test_size=0.2,
                                                      stratify=tgt_y,
                                                      random_state=args.seed)

    if args.model == 'roberta':
        src_features = roberta_convert_examples_to_features(src_x, src_y, args.max_seq_length, tokenizer)
        src_test_features = roberta_convert_examples_to_features(src_test_x, src_test_y, args.max_seq_length, tokenizer)
        tgt_features = roberta_convert_examples_to_features(tgt_train_x, tgt_train_y, args.max_seq_length, tokenizer)
        tgt_all_features = roberta_convert_examples_to_features(tgt_x, tgt_y, args.max_seq_length, tokenizer)
    else:
        src_features = convert_examples_to_features(src_x, src_y, args.max_seq_length, tokenizer)
        src_test_features = convert_examples_to_features(src_test_x, src_test_y, args.max_seq_length, tokenizer)
        tgt_features = convert_examples_to_features(tgt_train_x, tgt_train_y, args.max_seq_length, tokenizer)
        tgt_all_features = convert_examples_to_features(tgt_x, tgt_y, args.max_seq_length, tokenizer)

    # load dataset

    src_data_loader = get_data_loader(src_features, args.batch_size)
    src_data_loader_eval = get_data_loader(src_test_features, args.batch_size)
    tgt_data_loader = get_data_loader(tgt_features, args.batch_size)
    tgt_data_loader_all = get_data_loader(tgt_all_features, args.batch_size)

    # load models
    if args.model == 'bert':
        encoder = BertEncoder()
        cls_classifier = BertClassifier()
        dom_classifier = DomainClassifier()
    elif args.model == 'distilbert':
        encoder = DistilBertEncoder()
        cls_classifier = BertClassifier()
        dom_classifier = DomainClassifier()
    else:
        encoder = RobertaEncoder()
        cls_classifier = RobertaClassifier()
        dom_classifier = RobertaDomainClassifier()

    if args.load:
        encoder = init_model(encoder, restore=param.encoder_path)
        cls_classifier = init_model(cls_classifier, restore=param.cls_classifier_path)
        dom_classifier = init_model(dom_classifier, restore=param.dom_classifier_path)
    else:
        encoder = init_model(encoder)
        cls_classifier = init_model(cls_classifier)
        dom_classifier = init_model(dom_classifier)

    print("=== Start Training ===")
    if args.train:
        encoder, cls_classifier, dom_classifier = train(args, encoder, cls_classifier, dom_classifier,
                                                        src_data_loader, src_data_loader_eval,
                                                        tgt_data_loader, tgt_data_loader_all)

    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> after training <<<")
    evaluate(encoder, cls_classifier, tgt_data_loader_all)


if __name__ == '__main__':
    main()
