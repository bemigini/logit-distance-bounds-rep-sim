"""

Example of two models which have small KL and
equal embeddings, but dissimilar(in the linear sense) unembeddings.


"""

import torch

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cross_decomposition import CCA

from examples.util import model_construction
from examples.util import likelihood




# Classification problem with 8 labels, y_1, ..., y_8. 
# Inputs from the first four labels only have one correct label. 
# Inputs from the last four labels are either 50/50 from y_5 and y_6
# or 50/50 from y_7 and y_8

rng = np.random.default_rng(0)
num_labels = 8

# Get angles for first four labels 
first_four_angles = np.expand_dims(np.array([0, np.pi/16, 2*np.pi/16, 3*np.pi/16]), 1)

first_four_vectors = model_construction.get_2dvectors_from_rad_and_length(first_four_angles, 5.5)

plt.scatter(first_four_vectors.squeeze()[:, 0], first_four_vectors.squeeze()[:, 1])
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.show()


m1_last_four_angles = np.expand_dims(np.array([10*np.pi/16, 16*np.pi/16, 20*np.pi/16, 26*np.pi/16]), 1)
m1_last_four_vectors = model_construction.get_2dvectors_from_rad_and_length(m1_last_four_angles, 5.5)

m1_g_vectors = np.concatenate((first_four_vectors, m1_last_four_vectors), axis=0)


# Visualise m1 unembeddings
for i in range(num_labels):
    plt.scatter(m1_g_vectors.squeeze()[i, 0], m1_g_vectors.squeeze()[i, 1])
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.show()



# Model two has y_5, y_6 swapped and y_7, y_8 swapped.
m2_g_vectors = m1_g_vectors.copy()
m2_g_vectors[4] = m1_g_vectors[5]
m2_g_vectors[5] = m1_g_vectors[4]
m2_g_vectors[6] = m1_g_vectors[7]
m2_g_vectors[7] = m1_g_vectors[6]


for i in range(num_labels):
    plt.scatter(m2_g_vectors.squeeze()[i, 0], m2_g_vectors.squeeze()[i, 1])
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.show()


# Make embeddings (these will be the same for both models)
embedding_main_angles = np.expand_dims(
    np.array(
        [0, np.pi/16, 2*np.pi/16, 3*np.pi/16, 13*np.pi/16, 13*np.pi/16, 23*np.pi/16, 23*np.pi/16]), 1)

embedding_angle_noise = rng.uniform(-np.pi/256, np.pi/256, (8, 500))
embedding_angles = embedding_main_angles + embedding_angle_noise

embedding_lengths = np.abs(rng.standard_normal((8, 500))) + 2

embedding_vectors = model_construction.get_2dvectors_from_rad_and_length(embedding_angles, embedding_lengths)

all_fx = np.concatenate(embedding_vectors, axis = 0)


plt.scatter(all_fx[:, 0], all_fx[:, 1], s=3, alpha=0.5)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.show()



# Plot all model 1
plt.scatter(all_fx[:, 0], all_fx[:, 1], s=3, alpha=0.5)
for i in range(m1_g_vectors.shape[0]):
    plt.scatter(m1_g_vectors.squeeze()[i, 0], m1_g_vectors.squeeze()[i, 1])
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.show()


# Plot all model 2
plt.scatter(all_fx[:, 0], all_fx[:, 1], s=3, alpha=0.5)
for i in range(m2_g_vectors.shape[0]):
    plt.scatter(m2_g_vectors.squeeze()[i, 0], m2_g_vectors.squeeze()[i, 1])
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.show()


# Calculate KL between the two models

# Log-likelihoods for model 1
m1_log_ps = []

for i in range(num_labels):
    m_1_current_label_log_p = likelihood.log_likelihood(
        embedding_vectors[i, :], m1_g_vectors[i], m1_g_vectors)
    m1_log_ps.append(m_1_current_label_log_p)

m1_flat_log_ps = np.array(m1_log_ps).flatten()

# Log-likelihoods for model 12
m2_log_ps = []

for i in range(num_labels):
    m_2_current_label_log_p = likelihood.log_likelihood(
        embedding_vectors[i, :], m2_g_vectors[i], m2_g_vectors)
    m2_log_ps.append(m_2_current_label_log_p)

m2_flat_log_ps = np.array(m2_log_ps).flatten()


kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
kl_div_m1_m2 = kl_loss(torch.tensor(m1_flat_log_ps), torch.tensor(m2_flat_log_ps))
print(kl_div_m1_m2) # 0.0020
kl_div_m2_m1 = kl_loss(torch.tensor(m2_flat_log_ps), torch.tensor(m1_flat_log_ps))
print(kl_div_m2_m1) # 0.0017

# So the KL divergences are very small 
# However, as we see below, the relationship between the unembeddings is quite non-linear


# Getting CCA scores between the unembeddings 
n_components = 2
cca = CCA(n_components=n_components, max_iter=1000)
cca.fit(m1_g_vectors.squeeze(), m2_g_vectors.squeeze())
score = cca.score(m1_g_vectors.squeeze(), m2_g_vectors.squeeze())
print(f'Model 1 vs model 2 unembeddings CCA score: {score}') 
# Model 1 vs model 2 unembeddings CCA score, 5,6 and 7,8 far apart: 0.33583735570713

X_c, Y_c = cca.transform(m1_g_vectors.squeeze(), m2_g_vectors.squeeze())
corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
mean_corr = np.mean(corrs)
print(f'Model 1 vs model 2 unembeddings mean correlation: {mean_corr}') 
# Model 1 vs model 2 unembeddings mean correlation, 5,6 and 7,8 far apart: 0.6737728432158235
