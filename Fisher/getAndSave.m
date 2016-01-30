function getAndSave(path, numberOfCluster)

bm = spinner(path);
[means, covariances, priors] = vl_gmm(bm', numberOfCluster);
[encodes, namers] = loadNotPicked(path, means, priors, covariances);

