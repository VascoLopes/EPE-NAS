import os
import time
import argparse
import random
import numpy as np
import math
import pandas as pd
import tabulate

from tqdm import trange
from statistics import mean
from scipy import stats
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import metric_learn
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='./datasets/cifar', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='./datasets/NAS-Bench-201-v1_0-e61699.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--evaluate_size', default=256, type=int)
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=1, type=int)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.optim as optim

from models import get_cell_based_tiny_net

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed+41)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

import torchvision.transforms as transforms
from datasets import get_datasets
from config_utils import load_config
from nas_201_api import NASBench201API as API
from nats_bench import create

def l1_norm(ti, tj):
    #return np.sum( np.absolute( ti.cpu().numpy()-tj.cpu().numpy() )) /tj.cpu().shape[0]
    return np.linalg.norm(ti.cpu().numpy()-tj.cpu().numpy(), ord=1)/tj.cpu().shape[0]

def l2_norm(ti, tj):
    #print(tj.cpu().shape[0])
    #return np.sqrt(np.sum( np.power(ti.cpu().numpy()-tj.cpu().numpy(),2) )) /tj.cpu().shape[0]
    return np.linalg.norm(ti.cpu().numpy()-tj.cpu().numpy())/tj.cpu().shape[0]

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_batch_jacobian(net, x, target, to, device, args=None):
    net.zero_grad()

    x.requires_grad_(True)

    
    _, y = net(x)
    #print(activation['InferCell'][0].reshape(-1).shape)
    
    '''
    idx = {}
    for i, label in enumerate(target):

        if label.item() in idx.keys():
            idx[label.item()].append(i)
        else:
            idx[label.item()] = [i]

    #net.cells[-1].register_forward_hook(get_activation('InferCell'))
    #net.lastact.register_forward_hook(get_activation('InferCell'))
    df_l1 = pd.DataFrame(columns=['0','1','2','3','4','5','6','7','8','9'])
    df_l2 = pd.DataFrame(columns=['0','1','2','3','4','5','6','7','8','9'])
    for i in range(0,10):
        l1norm = []
        l2norm = []
        for j in range(0,10):
            t = 0 if i != j else 1
            l1norm.append(l1_norm(activation["InferCell"][idx[i][0]].reshape(-1), activation["InferCell"][idx[j][t]].reshape(-1)))
            l2norm.append(l2_norm(activation["InferCell"][idx[i][0]].reshape(-1), activation["InferCell"][idx[j][t]].reshape(-1)))
        df_l1.loc[i] = l1norm
        df_l2.loc[i] = l2norm
    #print(df_l1.head)
    #print(df_l1.to_markdown()) 
    #print()
    #print(df_l2.to_markdown()) 
    
    print(f'l1 norm: class0 - class4 :{l1_norm(activation["InferCell"][idx[0][0]].reshape(-1), activation["InferCell"][idx[7][0]].reshape(-1))}')
    print(f'l1 norm: class0 - class8 :{l1_norm(activation["InferCell"][idx[0][0]].reshape(-1), activation["InferCell"][idx[3][0]].reshape(-1))}')
    print(f'l1 norm: class0 - class0 :{l1_norm(activation["InferCell"][idx[0][0]].reshape(-1), activation["InferCell"][idx[0][1]].reshape(-1))}')
    print(f'l1 norm: class8 - class8 :{l1_norm(activation["InferCell"][idx[8][0]].reshape(-1), activation["InferCell"][idx[8][1]].reshape(-1))}')
    print(f'l2 norm: class0 - class8 :{l2_norm(activation["InferCell"][idx[0][0]].reshape(-1), activation["InferCell"][idx[7][0]].reshape(-1))}')
    print(f'l2 norm: class0 - class4 :{l2_norm(activation["InferCell"][idx[0][0]].reshape(-1), activation["InferCell"][idx[3][0]].reshape(-1))}')
    print(f'l2 norm: class0 - class0 :{l2_norm(activation["InferCell"][idx[0][0]].reshape(-1), activation["InferCell"][idx[0][1]].reshape(-1))}')
    print(f'l2 norm: class8 - class8 :{l2_norm(activation["InferCell"][idx[8][0]].reshape(-1), activation["InferCell"][idx[8][1]].reshape(-1))}')
    '''
    
    #print(idx)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    #grad = np.sum(net.classifier.weight.grad.detach().cpu().numpy())
    #weights = net.classifier.weight

    #print(x)

    return jacob, target.detach()#, grad

def y(x):
    if x > 0.5:
        return np.sqrt(0.25/(1+(1/(0.5-x)*(1/(0.5-x)))))
    elif x < 0.5:
        return - np.sqrt(0.25/(1+(1/(0.5-x)*(1/(0.5-x)))))
    return 0

def sig(x):
    return 1.0/(1+np.exp(-x*1.5))

myfunc_vec = np.vectorize(sig)
def saeid_la(jacob, labels=None, n_classes=10):
    corrs = np.corrcoef(jacob)
    corrs = myfunc_vec(corrs)
    
    score = []
    #print("1")
    for value in range(1, 10):
        #print(value/10)
        #print((corrs<(value/10)).sum())
        score.append((corrs<(value/10)).sum())
    #print("2")
    #print(score)
    return score

def eval_saeid_la_score(scores):
    #print("3")
    scores = np.array(scores)
    indices = list(range(0,len(scores)))
    iterator = 0
    maximum_iterator = len(scores[0])
    #print("4")
    
    while True and iterator < maximum_iterator:
        #print("----")
        #print(scores[0,iterator])
        
        #print(scores[indices])
        maximum_value = np.max(scores[indices,iterator]) # Get maximum value in the current iterator [:, iterator]
        #print("5")
        
        indices = np.argwhere(scores[indices,iterator] == maximum_value) # Get all indices with max value
        #print("6")
        
        if indices.shape[0] == 1:
            break
        #print("7")
        iterator += 1
    #print(indices)
    return indices[0][0]


def eval_saeid(jacob, labels=None, n_classes=10):
    per_class={}
    for i, label in enumerate(labels[0]):
        if label in per_class:
            per_class[label] = np.vstack((per_class[label],jacob[i]))
        else:
            per_class[label] = jacob[i]

    #print("1")
    ind_corr_matrix= {}
    corrs_matrices_scores = []
    size = np.iinfo(np.int16).max
    for c in per_class.keys():
        corrs = np.corrcoef(per_class[c])
        try:
            if (corrs.shape[0] < size): #store the shape of the smaller corr. matrix
                size = corrs.shape[0]
        except:
            continue
        ind_corr_matrix[c] = corrs

    #print("2")

    #keys = ind_corr_matrix.keys()
    #for key in keys:
    #    indices = np.random.choice(ind_corr_matrix[key].shape[0], size, replace=False)
    #    B = ind_corr_matrix[key][indices]
    #    ind_corr_matrix[key] = B[:,indices]
        #per_class[c] = per_class[c][np.random.choice(per_class[c].shape[0], size, replace=False)]
    #print("3")

    differences = []
    keys = list(ind_corr_matrix.keys())
    for i,key in enumerate(keys):        
        for _,key2 in enumerate(keys[i:]):
            
            #calculate smallest size
            matrix_a = ind_corr_matrix[key]
            matrix_b = ind_corr_matrix[key2]
            if matrix_a.shape[0] != matrix_b.shape[0]:
                biggest_matrix = matrix_a if matrix_a.shape[0] > matrix_b.shape[0] else matrix_b
                smallest_matrix = matrix_a if matrix_a.shape[0] < matrix_b.shape[0] else matrix_b
                size = smallest_matrix.shape[0]
                difference = np.zeros((size,size))
                samples = 5
                for _ in range(0,samples):
                    indices = np.random.choice(biggest_matrix.shape[0], size, replace=False)
                    B = biggest_matrix[indices]
                    C = B[:,indices]
                    difference += C-smallest_matrix
                difference /= samples
            else:
                difference = ind_corr_matrix[key]-ind_corr_matrix[key2]
            differences.append(normalize(difference))
    #print("4")

    score = 0
    for matrix in differences:
        for row in matrix:
            score += sum(i for i in row if np.abs(i) >= 0.05)
            #row = np.abs(row)
            #for cell in row:
                #if cell >= 0.05:
                #    score += cell
    #print("6")


    return score

#spearman_weights = [-1^(i+j) for i in range(1,3073) for j in range(1,3073)]
def eval_score(jacob, labels=None, n_classes=10):
    '''
    k = 1e-5

    per_class={}
    for i, label in enumerate(labels[0]):
        if label in per_class:
            per_class[label] = np.vstack((per_class[label],jacob[i]))
        else:
            per_class[label] = jacob[i]

    ind_corr_matrix_score = {}
    corrs_matrices_scores = []
    for c in per_class.keys():
        #print(per_class[c])
        corrs = np.corrcoef(per_class[c])
        s = np.sum(np.log(abs(corrs)+k))
        # A and B)
        ind_corr_matrix_score[c] = s
        # # Corr. matrix of corr.matrices'
        #corrs_matrices_scores.append(s)

    # Corr. matrix of corr.matrices'
    #corrs = np.corrcoef(corrs_matrices_scores)
    #score = np.sum(np.log(abs(corrs)+k))
    
    score = 0
    for c in per_class.keys():
        # A)
        #for cj in per_class.keys():
        #    score += np.absolute(ind_corr_matrix_score[c]-ind_corr_matrix_score[cj])
        
        # B)
        score += np.absolute(ind_corr_matrix_score[c])
    '''
    
    
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    return v
    k = 1e-5
    score = -np.sum(np.log(v + k) + 1./(v + k))
    #score = -np.sum(np.log(abs(corrs)+k))
    '''
    #print(f'Made score: {(np.sum(np.log(corrs+1)))}')
    #print(f'Made score 1/: {1/(np.sum(np.log(corrs+1)))}')
    #Par_corr = -np.linalg.inv(corrs)
    #rho, _ = stats.spearmanr(jacob)
    #spearman = np.sum(rho)# + 1./(rho + k))#np.sum(rho)
    #spearman = np.average(rho)
    '''
    '''
    spearman=0
    #seems interesting
    for i in range(0,len(rho)-1):
        for j in (0,len(rho[0])-1):
            spearman += rho[i][j]*spearman_weights[i]
    '''
    #score = -np.sum(np.log(v + k) + 1./(v + k))
    return score#np.sum(np.log(v + k) + 1./(v + k))#, spearman#np.sum(np.log(corrs+1)), v[0], #np.sum(Par_corr)


def plot_scatter(x,y, title="Score"):
    # x -> acc
    # y -> score
    plt.clf()
    plt.plot(x, y, 'o', color='orange');
    #plt.ylim(-8500, 500)
    plt.yscale('log')
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Score - log-scale")
    plt.title(title)
    plt.show()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
THE_START = time.time()
api = API(args.api_loc)
api2 = create('/home/vasco/Downloads/nas-without-training-master/datasets/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)


os.makedirs(args.save_loc, exist_ok=True)

train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_loc, cutout=0)

if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'

else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'

if args.trainval:
    cifar_split = load_config('config_utils/cifar-split.txt', None, None)
    train_split, valid_split = cifar_split.train, cifar_split.valid
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               num_workers=0, pin_memory=True, sampler= torch.utils.data.sampler.SubsetRandomSampler(train_split))

else:
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=0, pin_memory=True)

times     = []
chosen    = []
acc       = []
val_acc   = []
topscores = []

dset = args.dataset if not args.trainval else 'cifar10-valid'

order_fn = np.nanargmax#np.nanargmin

runs = trange(args.n_runs, desc='')

indices = np.random.randint(0,15625,100) #sample x networks
scores = []
accs = []
time_to_train_networks = 0
start = time.time()
for arch in indices:
    config = api.get_net_config(arch, args.dataset)
    config['num_classes'] = 10

    network = get_cell_based_tiny_net(config)  # create the network from configuration
    network = network.to(device)
    
    # get network score
    # store score - performance

    data_iterator = iter(train_loader)
    x, target = next(data_iterator)
    x, target = x.to(device), target.to(device)
    
    jacobs = []
    targets = []
    iterations = np.int(np.ceil(args.evaluate_size/args.batch_size))
    for i in range(iterations):
        jacobs_batch, target = get_batch_jacobian(network, x, target, None, None)
        jacobs.append(jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().numpy())
        targets.append(target.cpu().numpy())
    jacobs = np.concatenate(jacobs, axis=0)
    if(jacobs.shape[0]>args.evaluate_size):
        jacobs = jacobs[0:args.evaluate_size, :]

    try:
        #s, made_score, par_corr, spearman = eval_score(jacobs, None)
        s = eval_score(jacobs, targets, None)
        info      = api.query_by_index(arch)
        #print(api.get_more_info(arch, dset, None, is_random=False))
        #info = api.query_meta_info_by_index(arch)
        #print (info.get_metrics(dset, acc_type))
        #print("--")
        #print(info.get_metrics(dset, acc_type))
        acc2      = (info.get_metrics(dset, acc_type)['accuracy'])
        time_to_train_networks += api2.get_cost_info(arch, dset, hp=200)['T-train@total']
    except Exception as e:
        print(e)
        s = np.nan
        continue
    scores.append(s)
    accs.append(acc2)
time_to_train_networks += time.time()-start #time taken for the SVM train process itself

regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
#regr = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
regr.fit(scores, accs)


# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

for N in runs:
    start = time.time()
    indices = np.random.randint(0,15625,args.n_samples)
    scores = []
    for arch in indices:

        data_iterator = iter(train_loader)
        x, target = next(data_iterator)
        x, target = x.to(device), target.to(device)

        config = api.get_net_config(arch, args.dataset)
        config['num_classes'] = 10

        network = get_cell_based_tiny_net(config)  # create the network from configuration
        network = network.to(device)

        #jacobs, labels, grad= get_batch_jacobian(network, x, target, 1, device, args)
        jacobs = []
        targets = []
        grads = []
        iterations = np.int(np.ceil(args.evaluate_size/args.batch_size))
        for i in range(iterations):
            jacobs_batch, target = get_batch_jacobian(network, x, target, None, None)
            jacobs.append(jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().numpy())
            targets.append(target.cpu().numpy())

        jacobs = np.concatenate(jacobs, axis=0)
        if(jacobs.shape[0]>args.evaluate_size):
            jacobs = jacobs[0:args.evaluate_size, :]

        try:
            s = eval_score(jacobs, targets, None)
        except Exception as e:
            print(e)
            s = 0
        try:
            s = np.array(s)
            scores.append(regr.predict(s.reshape(1,-1)))
        except Exception as e:
            #print (e)
            scores.append(np.nan)
    #print(f'max test acc:{np.max(accs)}')

    #best_arch = indices[order_fn(scores)]
    best_arch = indices[order_fn(scores)]
    info      = api.query_by_index(best_arch)
    topscores.append(scores[order_fn(scores)])
    chosen.append(best_arch)
    acc.append(info.get_metrics(dset, acc_type)['accuracy'])

    if not args.dataset == 'cifar10' or args.trainval:
        val_acc.append(info.get_metrics(dset, val_acc_type)['accuracy'])

    times.append(time.time()-start)
    runs.set_description(f"acc: {mean(acc if not args.trainval else val_acc):.2f}%")
    
    print(times)
    #plot_scatter(accs, scores, "Per class corr. matrix-B")
    #plot_scatter(accs, made_scores, "Looking directly at the first Eigenvalue")
    #plot_scatter(accs, spearmans, "Spearman Correlation")

print(val_acc)
print(f"Final mean test accuracy: {np.mean(acc)}")
if len(val_acc) > 1:
    print(f"Final mean validation accuracy: {np.mean(val_acc)}")

state = {'train_time' : time_to_train_networks,
         'accs': acc,
         'val_accs': val_acc,
         'chosen': chosen,
         'times': times,
         'topscores': topscores,
         }

dset = args.dataset if not args.trainval else 'cifar10-valid'
fname = f"{args.save_loc}/{dset}_{args.n_runs}_{args.n_samples}_{args.seed}.t7"
torch.save(state, fname)
