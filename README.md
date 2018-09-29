# HelloML
Hello Machine Learning. Hello future.

This is a simple Machine Learning demo:
1. Get data set from UCI Machine Learning repository.
2. Data visualization and data set property check.
3. Create validation date set. Split data set into two, 80% to train models and 20% as a validation data set.
4. Build 6 algorithms: Logistic Regression (LR), Linear Discriminant Analysis (LDA), K-Nearest Neighbors (KNN), Classification and Regression Trees (CART), Gaussian Naive Bayes (NB), and Support Vector Machines (SVM).
5. Run the 6 algorithms on data set and compare their result.
6. Employ KNN to make predications.

# Pre-requirement
```bash
$ pip install scipy numpy matplotlib pandas sklearn
```

# Troubleshooting
If you've installed Zoom in MacOS, you may encounter following errors:
```bash
python[63572:51581202] Error loading /Users/Shared/ZoomOutlookPlugin/zOutlookPlugin.bundle/Contents/MacOS/zOutlookPlugin:  dlopen(/Users/Shared/ZoomOutlookPlugin/zOutlookPlugin.bundle/Contents/MacOS/zOutlookPlugin, 265): no suitable image found.  Did find:
	/Users/Shared/ZoomOutlookPlugin/zOutlookPlugin.bundle/Contents/MacOS/zOutlookPlugin: mach-o, but wrong architecture
	/Users/Shared/ZoomOutlookPlugin/zOutlookPlugin.bundle/Contents/MacOS/zOutlookPlugin: mach-o, but wrong architecture
```

This happens when performing the data visualization step mentioned before. But it doesn't matter.
