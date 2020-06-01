# <NOTEBOOK_PATH> <OUTPUT_PATH>
PAPERMILL := papermill --cwd ./notebooks
# NBX := jupyter nbconvert --ExecutePreprocessor.timeout=-1 --execute --to notebook --inplace

.PHONY: clean-datasets clean-models clean-all mkdir-results models test all
.PHONY: mlp-clustering mlp-lesion mlp-double-lesion make mlp-plots

clean-datasets:
	rm -rf datasets

clean-models:
	rm -rf training_runs_dir models

clean-all: clean-datasets clean-models

mkdir-results:
	mkdir -p results

datasets:
	bash prepare_all.sh datasets

models:
	bash prepare_all.sh models

# Running `pytest src` causes to the weird `sacred` and `tensorflow` import error:
# ImportError: Error while finding loader for 'tensorflow' (<class 'ValueError'>: tensorflow.__spec__ is None)
# https://github.com/IDSIA/sacred/issues/493
test:
	pytest src/tests/test_lesion.py
	pytest src/tests/test_utils.py
	pytest src/tests/test_cnn.py
	pytest src/tests/test_spectral_clustering.py

mlp-clustering: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-clustering.ipynb ./notebooks/mlp-clustering.ipynb

mlp-clustering-stability: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-clustering-stability.ipynb ./notebooks/mlp-clustering-stability.ipynb

mlp-clustering-stability-n-clusters: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-clustering-stability-different-n_clusters.ipynb ./notebooks/mlp-clustering-stability-different-n_clusters-K=2.ipynb -p N_CLUSTERS 2
	$(PAPERMILL) ./notebooks/mlp-clustering-stability-different-n_clusters.ipynb ./notebooks/mlp-clustering-stability-different-n_clusters-K=7.ipynb -p N_CLUSTERS 7
	$(PAPERMILL) ./notebooks/mlp-clustering-stability-different-n_clusters.ipynb ./notebooks/mlp-clustering-stability-different-n_clusters-K=10.ipynb -p N_CLUSTERS 10
    
mlp-learning-curve: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-learning-curve.ipynb ./notebooks/mlp-learning-curve.ipynb

mlp-lesion: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-lesion-test.ipynb ./notebooks/mlp-lesion-test.ipynb

# Using 10 clusters
mlp-lesion-TEN: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-lesion-test-TEN.ipynb ./notebooks/mlp-lesion-test-TEN.ipynb

mlp-double-lesion: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-double-lesion-test.ipynb ./notebooks/mlp-double-lesion-test-MNIST.ipynb -p MODEL_TAG MNIST
	$(PAPERMILL) ./notebooks/mlp-double-lesion-test.ipynb ./notebooks/mlp-double-lesion-test-MNIST+DROPOUT.ipynb -p MODEL_TAG MNIST+DROPOUT
	$(PAPERMILL) ./notebooks/mlp-double-lesion-test.ipynb ./notebooks/mlp-double-lesion-test-FASHION.ipynb -p MODEL_TAG FASHION
	$(PAPERMILL) ./notebooks/mlp-double-lesion-test.ipynb ./notebooks/mlp-double-lesion-test-FASHION+DROPOUT.ipynb -p MODEL_TAG FASHION+DROPOUT

mlp-plots:
	$(PAPERMILL) ./notebooks/mlp-plots.ipynb ./notebooks/mlp-plots.ipynb

mlp-analysis: mlp-clustering mlp-clustering-stability mlp-clustering-stability-n-clusters mlp-learning-curve mlp-lesion mlp-double-lesion mlp-plots

all: datasets models test mlp-analysis
