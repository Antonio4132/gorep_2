import glob
import matplotlib.pyplot as plt

from scipy.io import mmread
from sklearn.decomposition import PCA

path = 'blood_downloads/*.mtx'
files = glob.glob(path)

matrices = []

print("Reading files...")

for file in files:
    file_name = file.split('/')[-1].split('.')[0]
    print(f"\t> Reading file {file}")
    matrix = mmread(file)
    matrices.append({
        'project_ID': file_name,
        'matrix': matrix
    })


projected_matrices = []

print("Principal component analysis...")

for matrix in matrices:
    print(f"\t> Doing PCA of {matrix['project_ID']}")

    pca = PCA(n_components=2)
    projected = pca.fit_transform(matrix['matrix'].toarray())

    matrix['PCA_projected'] = projected

print("Plotting 2 PCs...")

for matrix in matrices:
    print(f"\t> Plotting PC1 and PC2 of {matrix['project_ID']}")
    plt.scatter(matrix['PCA_projected'][:, 0], matrix['PCA_projected'][:, 1], alpha=0.5, label=matrix['project_ID'])

    plt.savefig('blood_downloads/' + matrix['project_ID'] + '.png')