import argparse

def parse():
    parser = argparse.ArgumentParser(description='PErsonalized Transformer for Explainable Recommendation (PETER)')
    parser.add_argument('--data_path', type=str,
                        default='../data/Amazon/Movies and TV/reviews.pickle',
                        help='path for loading the pickle data')
    parser.add_argument('--index_dir', type=str,
                        default='../data/Amazon/Movies and TV/1/',
                        help='load indexes')
    parser.add_argument('--emsize', type=int, default=512,
                        help='size of embeddings')
    parser.add_argument('--nhead', type=int, default=8,
                        help='the number of heads in the transformer')
    parser.add_argument('--nhid', type=int, default=2048,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=6,
                        help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--lr', type=float, default=1,
                        help='initial learning rate')
    parser.add_argument('--con_lr', type=float, default=5e-3,
                        help='contrast learning rate')
    parser.add_argument('--rat_lr', type=float, default=5e-3,
                        help='contrast learning rate')
    parser.add_argument('--ll_lr', type=float, default=5e-3,
                        help='contrast learning rate')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='report interval')
    parser.add_argument('--checkpoint', type=str, default='with_ru_uc/',
                        help='directory to save the final model')
    parser.add_argument('--outf', type=str, default='with_ru_uc',
                        help='output file for generated text')
    parser.add_argument('--vocab_size', type=int, default=20000,
                        help='keep the most frequent words in the dict')
    parser.add_argument('--endure_times', type=int, default=5,
                        help='the maximum endure times of loss increasing on validation')
    parser.add_argument('--rating_reg', type=float, default=0.1,
                        help='regularization on recommendation task')
    parser.add_argument('--con_reg', type=float, default=0.1,
                        help='regularization on context prediction task')
    parser.add_argument('--conself_reg', type=float, default=0.1,
                        help='regularization on context prediction task')
    parser.add_argument('--text_reg', type=float, default=1.0,
                        help='regularization on text generation task')
    parser.add_argument('--peter_mask', action='store_true', default=False,
                        help='True to use peter mask; Otherwise left-to-right mask')
    parser.add_argument('--use_feature', action='store_true', default=False,
                        help='False: no feature; True: use the feature')
    parser.add_argument('--words', type=int, default=15,
                        help='number of words to generate for each sample')
    parser.add_argument('--gamma', type=float, default=0.15,
                        help='number of words to generate for each sample')
    parser.add_argument('--lamda', type=int, default=190,
                        help='number of words to generate for each sample')
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--logfile', type=str,
                        help='temperature for loss function')

    args = parser.parse_args()

    return args