#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os, time, shutil, random, argparse
import tensorflow as tf
import numpy as np
from itertools import islice
from functools import reduce

from model_cost import build_network
from dataset import create_dataset
from instance_loader import InstanceLoader
from util import load_weights, save_weights

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def run_batch(sess, model, batch, batch_i, epoch_i, time_steps, train=False, verbose=True):

    EV, W, C, edges_mask, route_exists, n_vertices, n_edges = batch

    # Compute the number of problems
    n_problems = n_vertices.shape[0]

    # Define feed dict
    feed_dict = {
        model['EV']: EV,
        model['W']: W,
        model['C']: np.array([ C[sum(n_edges[:i])] for i in range(n_problems) ]),
        model['time_steps']: time_steps,
        model['route_exists']: route_exists,
        model['n_vertices']: n_vertices,
        model['n_edges']: n_edges
    }

    if train:
        outputs = [model['train_step'], model['loss'], model['predictions'], model['deviations']]
    else:
        outputs = [model['loss'], model['predictions'], model['deviations']]
    #end

    # Run model
    loss, predictions, deviations = sess.run(outputs, feed_dict = feed_dict)[-3:]

    if verbose:
        # Print stats
        print('{train_or_test} Epoch {epoch_i} Batch {batch_i}\t|\t(n,m,batch size)=({n},{m},{batch_size})\t|\t(Loss,Deviation)=({loss:.4f},{dev:.4f})\t|\tAvg. Prediction, # Predicted =({avg_pred:.4f},{count_pred:.1f})'.format(
            train_or_test = 'Train' if train else 'Test',
            epoch_i = epoch_i,
            batch_i = batch_i,
            loss = loss,
            dev = np.mean(deviations),
            n = np.sum(n_vertices),
            m = np.sum(n_edges),
            batch_size = n_vertices.shape[0],
            avg_pred = np.mean(np.round(predictions)),
            count_pred = np.sum(np.round(predictions))
            ),
            flush = True
        )
    #end

    return loss, np.mean(deviations), np.mean(predictions)
#end

def summarize_epoch(epoch_i, loss, dev, pred, train=False):
    print('{train_or_test} Epoch {epoch_i} Average\t|\t(Loss,Dev)=({loss:.4f},{dev:.4f})\t|\tAvg. Prediction, # Predicted=({avg_pred:.4f},{count_pred:.1f})'.format(
        train_or_test = 'Train' if train else 'Test',
        epoch_i = epoch_i,
        loss = np.mean(loss),
        dev = np.mean(dev),
        avg_pred = np.mean(pred),
        count_pred = np.sum(pred)
        ),
        flush = True
    )
#end

def ensure_datasets(batch_size, train_params, test_params):
    
    if not os.path.isdir('train'):
        print('Creating {} Train instances'.format(train_params['samples']), flush=True)
        create_dataset(
            train_params['n_min'], train_params['n_max'],
            train_params['conn_min'], train_params['conn_max'],
            bins=train_params['bins'],
            samples=train_params['samples'],
            path='train',
            dataset_type=train_params['dataset_type'])
    #end

    if not os.path.isdir('test'):
        print('Creating {} Test instances'.format(test_params['samples']), flush=True)
        create_dataset(
            test_params['n_min'], test_params['n_max'],
            test_params['conn_min'], test_params['conn_max'],
            bins=test_params['bins'],
            samples=test_params['samples'],
            path='test',
            dataset_type=train_params['dataset_type'])
    #end
#end

if __name__ == '__main__':
    
    # Define argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', default=64, type=int, help='Embedding size for vertices and edges')
    parser.add_argument('-timesteps', default=25, type=int, help='# Timesteps')
    parser.add_argument('-dev', default=0, type=float, help='Target cost deviation')
    parser.add_argument('-epochs', default=32, type=int, help='Training epochs')
    parser.add_argument('-batchsize', default=16, type=int, help='Batch size')
    parser.add_argument('-seed', type=int, default=42, help='RNG seed for Python, Numpy and Tensorflow')
    parser.add_argument('--load', const=True, default=False, action='store_const', help='Load model checkpoint?')
    parser.add_argument('--save', const=True, default=False, action='store_const', help='Save model?')
    parser.add_argument('-dataset_type', default='euc_2D', help='Which type of dataset? (euc_2D or random)')

    # Parse arguments from command line
    args = parser.parse_args()

    # Set RNG seed for Python, Numpy and Tensorflow
    random.seed(vars(args)['seed'])
    np.random.seed(vars(args)['seed'])
    tf.set_random_seed(vars(args)['seed'])

    # Setup parameters
    d                       = vars(args)['d']
    time_steps              = vars(args)['timesteps']
    target_cost_dev         = vars(args)['dev']
    epochs_n                = vars(args)['epochs']
    batch_size              = vars(args)['batchsize']
    load_checkpoints        = vars(args)['load']
    save_checkpoints        = vars(args)['save']

    train_params = {
        'n_min': 20,
        'n_max': 40,
        'conn_min': 1,
        'conn_max': 1,
        'bins': 10**6,
        'batches_per_epoch': 128,
        'samples': 2**20,
        'dataset_type': vars(args)['dataset_type']
    }

    test_params = {
        'n_min': train_params['n_min'],
        'n_max': train_params['n_max'],
        'conn_min': 1,
        'conn_max': 1,
        'bins': 10**6,
        'batches_per_epoch': 32,
        'samples': 1024,
        'dataset_type': vars(args)['dataset_type']
    }
    
    # Ensure that train and test datasets exist and create if inexistent
    ensure_datasets(batch_size, train_params, test_params)

    # Create train and test loaders
    train_loader    = InstanceLoader("train")
    test_loader     = InstanceLoader("test")

    # Build model
    print("Building model ...", flush=True)
    GNN = build_network(d)

    # Disallow GPU use
    config = tf.ConfigProto( device_count = {"GPU":0})
    with tf.Session(config=config) as sess:

        # Initialize global variables
        print("Initializing global variables ... ", flush=True)
        sess.run( tf.global_variables_initializer() )

        # Restore saved weights
        if load_checkpoints: load_weights(sess,'./cost-checkpoints-{target_cost_dev}'.format(target_cost_dev=target_cost_dev));

        with open('cost-log-{target_cost_dev}.dat'.format(target_cost_dev=target_cost_dev),'w') as logfile:
            # Run for a number of epochs
            for epoch_i in np.arange(epochs_n):

                train_loader.reset()
                test_loader.reset()

                train_stats = { k:np.zeros(train_params['batches_per_epoch']) for k in ['loss','dev','pred'] }
                test_stats = { k:np.zeros(test_params['batches_per_epoch']) for k in ['loss','dev','pred'] }

                print("Training model...", flush=True)
                for (batch_i, batch) in islice(enumerate(train_loader.get_batches(batch_size, target_cost_dev)), train_params['batches_per_epoch']):
                    train_stats['loss'][batch_i], train_stats['dev'][batch_i], train_stats['pred'][batch_i], = run_batch(sess, GNN, batch, batch_i, epoch_i, time_steps, train=True, verbose=True)
                #end
                summarize_epoch(epoch_i,train_stats['loss'],train_stats['dev'],train_stats['pred'],train=True)

                print("Testing model...", flush=True)
                for (batch_i, batch) in islice(enumerate(test_loader.get_batches(batch_size, target_cost_dev)), test_params['batches_per_epoch']):
                    test_stats['loss'][batch_i], test_stats['dev'][batch_i], test_stats['pred'][batch_i], = run_batch(sess, GNN, batch, batch_i, epoch_i, time_steps, train=False, verbose=True)
                #end
                summarize_epoch(epoch_i,train_stats['loss'],train_stats['dev'],train_stats['pred'],train=False)

                # Save weights
                savepath = './cost-checkpoints-{target_cost_dev}/epoch={epoch}'.format(target_cost_dev=target_cost_dev,epoch=100*np.ceil((epoch_i+1)/100))
                os.makedirs(savepath, exist_ok=True)
                if save_checkpoints: save_weights(sess, savepath);

                logfile.write('{epoch_i} {trloss} {trdev} {trpred} {tstloss} {tstdev} {tstpred}\n'.format(
                    
                    epoch_i = epoch_i,

                    trloss = np.mean(train_stats['loss']),
                    trdev = np.mean(train_stats['dev']),
                    trpred = np.mean(train_stats['pred']),

                    tstloss = np.mean(train_stats['loss']),
                    tstdev = np.mean(train_stats['dev']),
                    tstpred = np.mean(train_stats['pred']),
                    )
                )
                logfile.flush()
            #end
        #end
    #end
#end
