import numpy as np
from config import *
import sys
import copy
#import torch

def classify(relevanceMap,semanticMap,vector):
    predictClass=-1
    min_dist=float('inf')
    
    for certain_class,semanticVector in semanticMap.items():
        dist=np.sqrt(np.dot(np.dot(vector-semanticVector,relevanceMap[certain_class]),(vector-semanticVector).transpose()))
        if dist<min_dist:
            min_dist=dist
            predictClass=certain_class

    return predictClass

def classify_evol(relevanceMap,semanticMap,vector,coef,coef_unseen):
    predictClass=-1
    min_dist=float('inf')
    min_dist_I=float('inf')
    max_dist_I=float('inf')
    min_dist_known=float('inf')
    mean_dist=0
    # threshold=2*vector.shape[0]*vector.shape[0]
    ifSeen=False
    eyeMat=np.eye(semanticMap[0].shape[0])


    for certain_class in range(num_seen):
        semanticVector=semanticMap[certain_class]
        dist=np.sqrt(np.dot(np.dot(vector-semanticVector,relevanceMap[certain_class]),(vector-semanticVector).transpose()))
        dist_I=np.sqrt(np.dot(np.dot(vector-semanticVector,eyeMat),(vector-semanticVector).transpose()))

        if dist_I < min_dist_I:
            min_dist_I=dist_I

        if dist_I > max_dist_I:
            max_dist_I=dist_I

        mean_dist+=dist_I

        if dist<3*np.sqrt(vector.shape[0])*coef:
            ifSeen=True

    mean_dist/=num_seen

    if not ifSeen:
        # first unseen instance shows up
        if len(semanticMap.keys())==num_seen:
            return -1
        else:
            ifknown = False
            for certain_unseen_class in set(semanticMap.keys())-set(list(range(num_seen))):
                semanticVector=semanticMap[certain_unseen_class]
                dist=np.sqrt(np.dot(np.dot(vector-semanticVector,eyeMat),(vector-semanticVector).transpose()))

                # belongs to the already-identified unseen class
                #if dist <= coef_unseen * min_dist_I:
                if dist <= coef_unseen * (min_dist_I+mean_dist)/2:
                    ifknown=True
                    break

            if ifknown:
                for certain_unseen_class in set(semanticMap.keys())-set(list(range(num_seen))):
                    semanticVector=semanticMap[certain_unseen_class]
                    dist=np.sqrt(np.dot(np.dot(vector-semanticVector,eyeMat),(vector-semanticVector).transpose()))
                    if dist<min_dist_known:
                        min_dist_known=dist
                        predictClass=certain_unseen_class
            
    else:
        for certain_class in range(num_seen):
            semanticVector=semanticMap[certain_class]
            dist=np.sqrt(np.dot(np.dot(vector-semanticVector,relevanceMap[certain_class]),(vector-semanticVector).transpose()))
            if dist<min_dist:
                min_dist=dist
                predictClass=certain_class

    return predictClass

# include the one shot sample
def cal_acc(path_prefix):
    semanticMap = {}
    for certain_class in range(num_seen):
        semanticMap[certain_class] = np.load(path_prefix+'/semanticVector_{}.npy'.format(certain_class))

    with open(cluster_path,'w') as f:
        for distance in ['covMap','varAttrMap','sigmaMap']:
            relevancePath = path_prefix+'/{}_{}.npy'
            relevanceMap = {}
            for certain_class in range(num_seen):
                relevanceMap[certain_class] = np.load(relevancePath.format(distance,certain_class))
                if distance == 'covMap':
                    relevanceMap[certain_class]=np.linalg.pinv(relevanceMap[certain_class])

            resultlines = []

            conf = np.zeros([num_seen, num_seen])

            num_sample = 0

            for certain_class in range(num_seen):
                predict_semantic=np.load(path_prefix+'/featureVector_{}.npy'.format(certain_class))

                num_sample+=predict_semantic.shape[0]

                for i in range(0, predict_semantic.shape[0]):

                    j = certain_class
                    transform_semantic=[predict_semantic[i, :]]

                    k = classify(relevanceMap,semanticMap,transform_semantic[0])
                    if k in range(num_seen):
                        conf[j, k] = conf[j, k] + 1
                #for i in range(0, len(classes)):
                #    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
                resultlines.append("Accuracy(class:{}):{}\n".format(certain_class,conf[certain_class,certain_class]/predict_semantic.shape[0]))

            resultlines.append('Seen accuracy:{}\n'.format(np.trace(conf)/num_sample))

            f.writelines(resultlines)
    ###########################
    
    print('Prediction using cluster finished')


# include the one shot sample
def cal_acc_evol(path_prefix):
    origin_semanticMap = {}
    for certain_class in range(num_seen):
        origin_semanticMap[certain_class] = np.load(path_prefix+'/semanticVector_{}.npy'.format(certain_class))

    print('Using cluster plus mode')
    #####################
    with open(zero_shot_path,'w') as f:
        for distance in ['covMap','varAttrMap']:
            relevancePath = path_prefix+'/{}_{}.npy'
            relevanceMap = {}
            for certain_class in range(num_seen):
                relevanceMap[certain_class] = np.load(relevancePath.format(distance,certain_class))
                if distance == 'covMap':
                    relevanceMap[certain_class]=np.linalg.pinv(relevanceMap[certain_class])

            coef_unseen=1.1
            coef = 0.1
            while coef < 10 :
                resultlines=[]
                print('Distance {} with coefficient {}\n'.format(distance,coef))

                semanticMap = copy.deepcopy(origin_semanticMap)

                conf = np.zeros([num_seen, num_seen])

                resultlines.append('Distance {} with coefficient {}\n'.format(distance,coef))

                num_sample = 0
                num_unseen = 0

                for certain_class in range(num_seen):
                    predict_semantic=np.load(path_prefix+'/featureVector_{}.npy'.format(certain_class))

                    num_sample+=predict_semantic.shape[0]

                    for i in range(0, predict_semantic.shape[0]):

                        j = certain_class
                        transform_semantic=[predict_semantic[i, :]]

                        k = classify_evol(relevanceMap,semanticMap,transform_semantic[0],coef,coef_unseen)
                        if k in range(num_seen):
                            conf[j, k] = conf[j, k] + 1
                        elif k==-1:
                            num_unseen+=1
                    #for i in range(0, len(classes)):
                    #    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
                    resultlines.append("Accuracy(class:{}):{}\n".format(certain_class,conf[certain_class,certain_class]/predict_semantic.shape[0]))

                resultlines.append('Seen accuracy:{}\n'.format(np.trace(conf)/num_sample))

                seen_false=num_unseen/num_sample
                resultlines.append('Seen sample\'s unseen false rate:{}\n'.format(seen_false))

                print('Seen false rate:{}'.format(seen_false))

                if seen_false > 0.2:
                    coef+=0.2
                    #continue

                correct_unseen = 0
                total_unseen = 0
                new_class_total = {}
                new_class_correct = {}
                new_class_instances = {}

                new_class_index = num_seen
                new_class_map = {}

                class_len_map = {}

                for certain_class in range(num_seen,num_class_total):
                    predict_semantic=np.load(path_prefix+'/featureVector_{}.npy'.format(certain_class))

                    class_len_map[certain_class] = len(predict_semantic)

                    for i in range(0, predict_semantic.shape[0]):
                        transform_semantic=[predict_semantic[i, :]]

                        k = classify_evol(relevanceMap,semanticMap,transform_semantic[0],coef,coef_unseen)

                        total_unseen+=1

                        # TODO when k==-1, there's a new class identified 
                        if k==-1:
                            correct_unseen+=1
                            new_class_map[new_class_index]=certain_class
                            semanticMap[new_class_index]=transform_semantic[0]
                            new_class_instances[new_class_index]=[transform_semantic[0]]
                            new_class_correct[new_class_index]=1
                            new_class_total[new_class_index]=1
                            new_class_index+=1

                        elif not k in range(num_seen):
                            correct_unseen+=1
                            new_class_instances[k].append(transform_semantic[0])
                            semanticMap[k]=np.mean(new_class_instances[k],axis=0)
                            new_class_total[k]+=1
                            if new_class_map[k]==certain_class:
                                new_class_correct[k]+=1

                        else:
                            
                            print(k,certain_class)

                print(new_class_index)
                unseen_acc=correct_unseen/total_unseen
                resultlines.append('Unseen accuracy:{}\n'.format(unseen_acc))
                for certain_class in new_class_total.keys():
                    resultlines.append('Accuracy for newly-identified unseen class {}(origin class {}):{}\n'.format(certain_class,new_class_map[certain_class],new_class_correct[certain_class]/class_len_map[new_class_map[certain_class]]))
               
                print('unseen total {},unseen correct {}'.format(total_unseen,correct_unseen))
                print(seen_false,unseen_acc)
                if unseen_acc > 0.8:
                    f.writelines(resultlines)

                if coef<1.0:
                    coef+=0.02
                else:
                    coef+=0.5
                if unseen_acc < 0.8:
                    break
    
    print('{} cluster finished'.format('Evalution zero-shot'))

if __name__ == '__main__':

    evol = True
    if len(sys.argv) > 1 and sys.argv[1] == '-c':
        evol = False

    model_paths = []
    if len(sys.argv) > 2 :
        path = sys.argv[2]
        if os.path.isdir(path):
            model_paths = [os.path.join(path,x) for x in os.listdir(path)]
        elif os.path.isfile(path):
            model_paths.append(path)
    else:
        model_paths.append(model_path)

    if not os.path.isdir('result_0_shot/'+modification):
        os.mkdir('result_0_shot/'+modification)

    for model_path in model_paths:
        print('Loading semantic from model {}'.format(model_path))
        path_prefix='feature/'+modification+'/'+os.path.split(model_path)[-1][:-4]

        zero_shot_path = 'result_0_shot/'+modification+'/acc_'+os.path.split(model_path)[-1][:-4]+'.txt'
        cluster_path = 'result_0_shot/'+modification+'/cluster_'+os.path.split(model_path)[-1][:-4]+'.txt'
        #print('With {} epochs training...'.format(checkpoint['epoch']))
        
        if evol:
            print('Using evolution zero-shot mode')
            cal_acc_evol(path_prefix)
        else:
            print('Using normal zero-shot mode')
            cal_acc(path_prefix)
            
print('end')
