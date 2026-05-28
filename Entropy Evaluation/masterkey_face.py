''' This script uses CMA-ES to find masterkeys in feature space'''
import numpy as np
import cma
import scipy
from sklearn.decomposition import PCA

from spherical_codes import generate_spherical_coord_uniform
import guessing_metrics as em

#import tensorflow as tf

def objective_func_discr(master_key, centroids, pca, thr):
    ''' This is the objective function to optimize
    argmin( 1/n sum( (n-cos_dist(c_i, master_key))))
    Args
        @param master_key the guess from the optimizer
        @param centroids, the users to be impersonated
        @param thr, the threshold on distance for impersonation
        @returns the score
    '''
    n = centroids.shape[0]
    master_key = pca.inverse_transform(master_key)
    # normalize the master_key
    master_key = master_key/np.linalg.norm(master_key)
    #dist = scipy.spatial.distance.cdist(centroids, master_key, metric='cosine')
    dist = 1-np.matmul(centroids, master_key.T)
    nb_centroids_matched = np.sum(dist < thr)
    return (n-nb_centroids_matched)/n

if __name__ == '__main__':

    num_enrolled_users = 10000

    # determine frr and far
    # load thresholds
    thrs = np.load('../../../data/templates/face_templates/thresholds.npy')
    thr = thrs[0]
    x0 = generate_spherical_coord_uniform(512,1)
    #x0 = np.array([0]*512)

    # load centroids
    centroids = np.load('../../../data/templates/face_templates/centroids.npz')['x'][:num_enrolled_users]
    # PCA transform to work in lower dimensional state.
    pca = PCA(n_components=64)
    pca.fit(centroids)
    total_users = centroids.shape[0]
    options = {'bounds': [[-4]*64, [4]*64]}
    cma_keys = []
    prob_list = []


    # use cma to find master keys
    while centroids.shape[0] > total_users*0.5:
        #x0 = generate_spherical_coord_uniform(64,1)
        x0 = np.array([[0]*64])
        master_key = cma.fmin2(lambda x: objective_func_discr([x], centroids, pca, thr), x0, 1, options, restarts=2)[0]

        # filter out centroids broken by master_key
        master_key = pca.inverse_transform(master_key)
        master_key = master_key/np.linalg.norm(master_key)
        cma_keys.append(master_key)
        dist = 1-np.matmul(centroids, np.array([master_key]).T)

        nb_centroids_matched = np.sum(dist < thr)
        dist = dist.reshape((dist.shape[0],))
        print(nb_centroids_matched)
        centroids = centroids[np.logical_not(dist < thr)]
        print(centroids.shape)

        # add success rate to prob_list
        prob_list.append(nb_centroids_matched/total_users)

    min_entr = em.min_entropy(prob_list[0])
    guess_work_entropy25 = em.guess_work_entropy(0.25, prob_list)
    guess_work_entropy50 = em.guess_work_entropy(0.5, prob_list)

    beta3 = em.beta_entropy(3, prob_list)
    beta5 = em.beta_entropy(5, prob_list)
    beta10 = em.beta_entropy(10, prob_list)

    print(np.round(min_entr, 2), '&', np.round(beta3,2), '&', np.round(beta5,2), '&',
          np.round(beta10,2), '&', np.round(guess_work_entropy25,2), '&',
          np.round(guess_work_entropy50,2) )





