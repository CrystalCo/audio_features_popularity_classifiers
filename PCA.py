from sklearn.decomposition import PCA as PCA

def PCATransform(data):
    pca = PCA()
    pca.fit(data)

    toKeep = 0
    totalVar = 0
    # Find PC's that contribute at least 95% data variance
    for var in pca.explained_variance_ratio_:
        if totalVar < .95:
            totalVar += var
            toKeep += 1
        else:
            break

    print("Top {} components capture {:.2f}% of the data".format(toKeep,totalVar*100))
    
    # Transform data and select only 'toKeep' PC's, convert back to pandas dataframe
    outData = PCA(n_components = toKeep).fit_transform(data)
    outData = pd.DataFrame(data=outData)
    print("{} -> {}".format(data.shape,outData.shape))
    
    return outData