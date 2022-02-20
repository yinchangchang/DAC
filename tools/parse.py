# coding=utf8

import argparse

parser = argparse.ArgumentParser(description='DAC')

parser.add_argument(
        '--data-dir',
        type=str,
        default='../data/',
        help='data files directory'
        )
parser.add_argument(
        '--file-dir',
        type=str,
        default='../file/',
        help='files directory'
        )
parser.add_argument(
        '--result-dir',
        type=str,
        default='../result/',
        help='result files directory'
        )
parser.add_argument( '--resume', metavar='resume saved model', type=str, default='')
parser.add_argument( '--resume-i', metavar='resume saved model', type=str, default='')
parser.add_argument(
        '--gpu',
        type=int,
        default=0
        )


parser.add_argument(
        '--dataset',
        type=str,
        # default='mimic_json',
        default='EHR_json',
        help='mimic_json or EHR_json'
        )
parser.add_argument(
        '--phase',
        '-p',
        metavar='PHASE',
        type=str,
        default='train',
        help='train or test'
        )
parser.add_argument(
        '--batch-size',
        '-b',
        metavar='BATCH SIZE',
        type=int,
        default=256,
        help='batch size'
        )
parser.add_argument('-j',
        '--workers',
        default=16,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 32)')
parser.add_argument('--lr',
        '--learning-rate',
        default=0.0001,
        type=float,
        metavar='LR',
        help='initial learning rate')
parser.add_argument('--loss', default='no', type=str)
parser.add_argument('--autoencoder-epochs', default=50, type=int)
parser.add_argument('--action-imitation-epochs', default=1000, type=int)
parser.add_argument('--state-prediction-epochs', default=1000, type=int)
parser.add_argument('--finetune-epochs', default=20, type=int)

parser.add_argument('--save-freq',
        default=10,
        type=int,
        metavar='S',
        help='save frequency')
parser.add_argument('--val-freq',
        default=1,
        type=int,
        metavar='S',
        help='val frequency')

parser.add_argument( '--use-ci', default=2, type=int)
parser.add_argument(
        '--use-cat',
        default=0,
        type=int,
        help='use concatenation'
        )
parser.add_argument(
        '--use-gp',
        default=1,
        type=int,
        help='use golabl pooling in FCModel'
        )
parser.add_argument(
        '--use-aug',
        default=1,
        type=int,
        help='use data augmentation'
        )
parser.add_argument(
        '--use-cui',
        default=0,
        type=int,
        help='map ehr id into cui id'
        )
parser.add_argument(
        '--use-xt',
        default=0,
        type=int,
        help='use xt'
        )
parser.add_argument(
        '--xt-output',
        default=0,
        type=int,
        help='use xt as output'
        )
parser.add_argument(
        '--use-vs-att',
        default=0,
        type=int,
        help='use visit attention'
        )

# RNN
parser.add_argument(
        '--num-layers',
        default=1,
        type=int,
        help='num-layers'
        )
parser.add_argument(
        '--hidden-size',
        default=512,
        type=int,
        help='--hidden-size'
        )
parser.add_argument(
        '--embed-size',
        default=512,
        type=int,
        help='--embed-size'
        )
parser.add_argument(
        '--rnn-type',
        default='lstm',
        type=str,
        help='--'
        )
parser.add_argument(
        '--rnn-size',
        default=512,
        type=int,
        help='--'
        )
parser.add_argument(
        '--drop-prob-lm',
        default=0.5,
        type=float,
        help='--'
        )
parser.add_argument(
        '--seq-length',
        default=120,
        type=int,
        help='--'
        )

parser.add_argument(
        '--predict-day',
        default=7,
        type=int,
        help='predict whether the patient will be diagnosed with HF after N days'
        )
parser.add_argument(
        '--nc',
        default=1000,
        type=int,
        help='number of case patients that are used in the training process'
        )
parser.add_argument(
        '--words',
        default='',
        type=str,
        help='--'
        )
parser.add_argument(
        '--hard-mining',
        default=0,
        type=int,
        help='use hard mining'
        )
# RETAIN

parser.add_argument(
        '--dim-alpha',
        default=512,
        type=int,
        help='--dim-alpha'
        )

parser.add_argument(
        '--dim-beta',
        default=512,
        type=int,
        help='--dim-beta'
        )
parser.add_argument(
        '--dropout-context',
        default=0.5,
        type=float,
        help='--dropout-context'
        )



args = parser.parse_args()
