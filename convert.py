#!/usr/bin/env python

from kaffe import KaffeError, print_stderr
from kaffe.tensorflow import TensorFlowTransformer

import numpy as np
import argparse


def fatal_error(msg):
    print_stderr(msg)
    exit(-1)


def convert(def_path, caffemodel_path, data_output_path, code_output_path, phase):
    try:
        transformer = TensorFlowTransformer(def_path, caffemodel_path, phase=phase)
        print_stderr('Converting data...')
        if caffemodel_path is not None:
            data = transformer.transform_data()
            print_stderr('Saving data...')
            with open(data_output_path, 'wb') as data_out:
                np.save(data_out, data)
        if code_output_path:
            print_stderr('Saving source...')
            with open(code_output_path, 'wb') as src_out:
                src_out.write(transformer.transform_source())
        print_stderr('Done.')
    except KaffeError as err:
        fatal_error('Error encountered: {}'.format(err))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototxt', required=True, help='Model definition (.prototxt) path')
    parser.add_argument('--caffemodel', help='Model data (.caffemodel) path')
    parser.add_argument('--data-output', default='weights.npy', help='Converted data output path')
    parser.add_argument('--code-output', default='model.py', help='Save generated source to this path')
    parser.add_argument('--phase', default='test', help='The phase to convert: test (default) or train')
    args = parser.parse_args()
    convert(args.prototxt, args.caffemodel, args.data_output, args.code_output, args.phase)


if __name__ == '__main__':
    main()
