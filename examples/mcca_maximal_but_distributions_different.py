"""

Example of how two models, (f, g), (f', g'), can have maximal mean canonical correlation
between both embeddings, mCCA(f(x), f'(x)) = 1, and unembeddings, mCCA(g(y), g'(y)) = 1,
but the distributions of the models are different. 


"""



import torch

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cross_decomposition import CCA

from examples.util import model_construction
from examples.util import likelihood




# Classification problem with 5 labels, y_1, ..., y_5. 

rng = np.random.default_rng(0)
num_labels = 5
colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:cyan', 'tab:purple']


# Get angles for the unembeddings for the first model, m1. 
m1_unemb_angles = np.expand_dims(np.array([0, np.pi/2, np.pi, 5*np.pi/4, 7*np.pi/4]), 1)

m1_unembs = model_construction.get_2dvectors_from_rad_and_length(m1_unemb_angles, 8)


fig, ax = plt.subplots(1, 1, figsize=(6, 5))
for i in range(num_labels):
    ax.scatter(m1_unembs.squeeze()[i, 0], m1_unembs.squeeze()[i, 1], s=200, c = colours[i], label = f'{i+1}')
ax.set_xlim(-9, 9)
ax.set_ylim(-9, 9)
ax.legend()
fig.show()


# The embeddings are in clusters at the same angles as the unembeddings
embedding_clusters = rng.normal(0, 0.5, size = (5, 100, 2))
emb_directions = model_construction.get_2dvectors_from_rad_and_length(m1_unemb_angles, 5)
embedding_clusters = embedding_clusters + emb_directions

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
for i in range(num_labels):
    embs = embedding_clusters[i]
    ax.scatter(embs[:, 0], embs[:, 1], c = colours[i], s=20, alpha = 0.7, label = f'{i+1}')
ax.set_xlim(-9, 9)
ax.set_ylim(-9, 9)
ax.legend()
fig.show()


# We will make model 2, m2, by rotating the embeddings and unembeddings in opposite directions.
rotation_counter_clockwise = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])
rotation_clockwise = np.array([[np.cos(np.pi/4), np.sin(np.pi/4)], [-np.sin(np.pi/4), np.cos(np.pi/4)]])


rotated_m1_unembs = np.matmul(np.expand_dims(rotation_clockwise, axis=0), np.transpose(m1_unembs, axes=(0, 2, 1)))
m2_unembs = rotated_m1_unembs

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
for i in range(num_labels):
    ax.scatter(m2_unembs.squeeze()[i, 0], m2_unembs.squeeze()[i, 1], s=200, c = colours[i], label = f'{i+1}')
ax.set_xlim(-9, 9)
ax.set_ylim(-9, 9)
ax.legend()
fig.show()


rotated_m1_embs = np.matmul(np.expand_dims(rotation_counter_clockwise, axis=0), np.transpose(embedding_clusters, axes=(0, 2, 1)))
m2_embs = np.transpose(rotated_m1_embs, axes=(0, 2, 1))

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
for i in range(num_labels):
    embs = m2_embs[i]
    ax.scatter(embs[:, 0], embs[:, 1], c = colours[i], s=20, alpha = 0.7, label = f'{i+1}')
ax.set_xlim(-9, 9)
ax.set_ylim(-9, 9)
ax.legend()
fig.show()


# Plotting complete model 1
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
for i in range(num_labels):
    embs = embedding_clusters[i]
    ax.scatter(m1_unembs.squeeze()[i, 0], m1_unembs.squeeze()[i, 1], s=200, c = colours[i], label = f'{i+1}')
    ax.scatter(embs[:, 0], embs[:, 1], c = colours[i], s=20, alpha = 0.7)
ax.set_xlim(-9, 9)
ax.set_ylim(-9, 9)
ax.set_title('model 1', fontsize = 20)
ax.legend()
fig.show()


# Plotting complete model 2
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
for i in range(num_labels):
    embs = m2_embs[i]
    ax.scatter(m2_unembs.squeeze()[i, 0], m2_unembs.squeeze()[i, 1], s=200, c = colours[i], label = f'{i+1}')
    ax.scatter(embs[:, 0], embs[:, 1], c = colours[i], s=20, alpha = 0.7)
ax.set_xlim(-9, 9)
ax.set_ylim(-9, 9)
ax.set_title('model 2', fontsize = 20)
ax.legend()
fig.show()

# So now embeddings of model 1 are a rotation of embeddings of model 2
# and unembeddings of model 1 are also a rotation of unembeddings of model 2,
# but the distributions of these models will be very different.
# For example, the embeddings which model 1 will assign label 1, will be 
# assigned label 2 by model 2. 



# Calculating mean canonical correlation 
# (this will be 1 by construction, but there might be numerical inaccuracies)
n_components = 2
cca = CCA(n_components=n_components, max_iter=1000)
cca.fit(m1_unembs.squeeze(), m2_unembs.squeeze())
score = cca.score(m1_unembs.squeeze(), m2_unembs.squeeze())
print(f'Model 1 vs model 2 unembeddings CCA score: {score}')
# Model 1 vs model 2 unembeddings CCA score: 0.9997594629675379

X_c, Y_c = cca.transform(m1_unembs.squeeze(), m2_unembs.squeeze())
corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
mean_corr = np.mean(corrs)
print(f'Model 1 vs model 2 unembeddings mean correlation: {mean_corr}') 
# Model 1 vs model 2 unembeddings mean canonical correlation: 1.0


n_components = 2
m1_embs = np.concatenate(embedding_clusters, axis = 0)
m2_embs = np.concatenate(m2_embs, axis = 0)
cca = CCA(n_components=n_components, max_iter=1000)
cca.fit(m1_embs, m2_embs)
score = cca.score(m1_embs, m2_embs)
print(f'Model 1 vs model 2 embeddings CCA score: {score}') 
# Model 1 vs model 2 embeddings CCA score: 0.9998188610750391

X_c, Y_c = cca.transform(m1_embs, m2_embs)
corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
mean_corr = np.mean(corrs)
print(f'Model 1 vs model 2 embeddings mean correlation: {mean_corr}')
# Model 1 vs model 2 embeddings mean canonical correlation: 0.9999999999999998
