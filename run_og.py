import pandas as pd
import argparse
from ginnlp.ginnlp import GINNLP

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--format', type=str, default='csv', help='Format of dataset file')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--round_digits', type=int, default=3, help='Number of digits to round to')
    parser.add_argument('--start_ln_blocks', type=int, default=1, help='Number of starting blocks')
    parser.add_argument('--growth_steps', type=int, default=3, help='Number of growth steps')
    parser.add_argument('--l1_reg', type=float, default=1e-4, help='L1 regularization')
    parser.add_argument('--l2_reg', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--init_lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--decay_steps', type=int, default=1000, help='Decay steps')
    parser.add_argument('--reg_change', type=float, default=0.5,
                        help='Fraction of epochs to regularization change')
    parser.add_argument('--train_iter', type=int, default=4, help='Number of training iterations')
    args = parser.parse_args()
    data_file = args.data
    num_epochs = args.num_epochs
    round_digits = args.round_digits
    start_ln_blocks = args.start_ln_blocks
    growth_steps = args.growth_steps
    l1_reg = args.l1_reg
    l2_reg = args.l2_reg
    init_lr = args.init_lr
    decay_steps = args.decay_steps
    reg_change = args.reg_change
    train_iter = args.train_iter

    if args.format == 'csv':
        sep = ','
    elif args.format == 'tsv':
        sep = '\t'
    else:
        raise ValueError('Invalid format')

    df = pd.read_csv(data_file, sep=sep)
    print(df.shape)
    df = df.sample(10000)
    train_x = df.drop('target', axis=1).values.astype(float)
    train_y = df['target'].values

    model = GINNLP(num_epochs=num_epochs, round_digits=round_digits, start_ln_blocks=start_ln_blocks,
                   growth_steps=growth_steps, l1_reg=l1_reg, l2_reg=l2_reg,
                   init_lr=init_lr, decay_steps=decay_steps, reg_change=reg_change, train_iter=train_iter)
    print('Fitting GINN-LP')
    model.fit(train_x, train_y)
    print('recovered equation: ', model.recovered_eq)