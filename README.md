# sequence-to-sequence

by using pytorch, lstm, multi-gpu to translation


## Arguments
``` bash
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        metavar='N',
                        help='number of total epochs to run')
```


## References 
https://github.com/bentrevett/pytorch-seq2seq
