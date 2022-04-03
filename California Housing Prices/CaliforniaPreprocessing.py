from joblib import load

class Preprocessing():
    
    def __init__(self):
        
        #Load various trained models
        self.latmodel = load("models/gmmlat.joblib")
        self.longmodel = load("models/gmmlong.joblib")
        self.scaler = load("models/standardscaler.joblib")
        self.linear_model = load("models/linearmodel.joblib")
    
    def transform(self, X):
        """
        Transforms the given feature matrix with a series of operations performed on the training set. 
        """
        
        X_ohe = self.onehotencoding(X)
        X_gmm = self.gaussian_mixture_prep(X,self.latmodel, self.longmodel)
        X_feat = self.feature_combination(X)
        X_std = self.scaler.transform(X_feat)
        return X_std
        
    
    def gaussian_mixture_prep(X, gmmlat, gmmlong):

        """
        Creates new binary features by assigning cluster labels to latitude and longitude feature values calculated by 
        gaussian mixture model. 
        
        @params:
        gmmlat: gaussian mixture model for latitude
        gmmlong: gaussian mixture model for longitude
        
        """

        X['latitude_class'] = gmmlat.predict(X['latitude'].values.reshape(-1, 1))
        X['longitude_class'] = gmmlong.predict(X['longitude'].values.reshape(-1, 1))
        X['latitude_class'] = X['latitude_class'].astype(str)
        X['longitude_class'] = X['longitude_class'].astype(str)
        X['latlongcluster'] = X['latitude_class'] + X['longitude_class']
        X = X.drop(columns=['latitude_class', 'longitude_class'])
        X= pd.get_dummies(X, columns=['latlongcluster'])
        X.drop(columns=['latlongcluster_00'], inplace=True)
    
    return X

    def latlongtoxyz(X):
        
        """
        Creates new features by transforming latitude and longitude to x,y,z coordinates
        """
        
        X['xcoordinate'] = np.cos(X['latitude'])*np.cos(X['longitude'])
        X['ycoordinate'] = np.cos(X['latitude'])*np.sin(X['longitude'])
        X['zcoordinate'] = np.sin(X['latitude'])
        X = X.drop(columns = ['latitude','longitude'])
        return X
        
    def feature_combination(X):
        
        """
        creates new features using total_rooms, total_bedrooms, households and populatiom
        """
        
        X['total_rooms_per_person'] = X['total_rooms']/X['population']
        X['total_bedrooms_per_person'] = X['total_bedrooms']/X['population']
        X['households_per_person'] = X['households']/X['population']
        
        return X
                           
    def onehotencoding(X):
        
        """
        performs dummy encoding for the ocean proximity feature
        """
        
        X = pd.get_dummies(X, columns=['ocean_proximity'])
        X.drop(columns=['ocean_proximity_ISLAND'], inplace=True)
        return X
                           
    
    def predict(X):
        
        """
        provides predictions y for a given X
        """
        
        self.y_pred = self.linear_model.predict(X)
        return self.y_pred
                           