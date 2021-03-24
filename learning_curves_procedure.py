# Time series
import numpy as np 
import xgboost as xgb
import matplotlib.pyplot as plt
import statistics as sts
import datetime
import math

def mean_absolute_percentage_error(y_true, y_pred):
    """     
    Parameters
    ----------
    y_true : array-like 
        the vector of actual values of target variable
    y_pred : array-like 
        the vector of predicted values of target variable

    Returns
    ----------
    mape: float
        the mean absolute percentage error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    try:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except:
        mape = np.NaN
    return mape

def method(X_train, X_test, Y_train, Y_test):
    """     
    Parameters
    ----------
    X_train : Dataframe 
        all the independent variables of training-set
    X_test : Dataframe 
        all the independent variables of test-set
    Y_train : Dataframe 
        the dependent variable of training-set. Column name 'target'
    Y_test : Dataframe 
        the dependent variable of test-set. Column name 'target'

    Returns
    ----------
    Yhat_train : array-like 
        the vector of predicted values of target variable of training-set
    Yhat_test : array-like 
        the vector of predicted values of target variable of test-set
    """
    # Any other model from sklearn can be used
    rs = xgb.XGBRegressor(max_depth=4, learning_rate=0.1, n_estimators=10, silent=True, objective='reg:linear', reg_lambda=1)

    rs.fit(X_train,Y_train) 

    Yhat_train=rs.predict(X_train)
    Yhat_test=rs.predict(X_test)

    return Yhat_train, Yhat_test


def plot_learning_curves(X, Y, d, train_sizes, k):
    """ 
    Generate a plot: the test and training learning curve.

    Parameters
    ----------
    X : Dataframe
        all the independent variables of dataset

    Y : Dataframe
        the target variable of dataset. Column name 'target'

    d : float
        the test-set size in days

    train_sizes : list
        a list of q elements. Each element p_j is a training-set size in weeks

    k : int
        number of tested days
    """
    train_mape_curve, test_mape_curve = [],[]
    tr_mape_std, ts_mape_std, tr_mape, ts_mape = [],[],[],[]

    # Learning curve procedure
    for tr in train_sizes: # for each training-set size in the list
        for i in range(k): # repeat the preocedure for k different days
            
            day_first = X.index[0] + datetime.timedelta(weeks=tr)
            timedelta = ((X.index[-1] - day_first)/k).days
            day_ts = day_first + datetime.timedelta(days=timedelta*i)

            sp_tr = day_ts - datetime.timedelta(weeks=tr) # First index of training-set
            ep_tr = day_ts  - datetime.timedelta(minutes=1) # Last index of training-set
            sp_ts = day_ts # First index of test-set
            # Last index of test-set: subtract one minute to not select the next day
            ep_ts = day_ts + datetime.timedelta(days = d) - datetime.timedelta(minutes=1)
            
            if (ep_ts) > X.index[-1]:
                break

            # Build training-set & test-set
            X_train = X[sp_tr:ep_tr]
            Y_train = Y[sp_tr:ep_tr]
            X_test = X[sp_ts:ep_ts]
            Y_test = Y[sp_ts:ep_ts]

            # Forecast
            Yhat_train, Yhat_test = method(X_train, X_test, Y_train, Y_test)

            # Test and train error metrics
            mape_train = mean_absolute_percentage_error(np.array(Y_train.target), Yhat_train)
            mape_test = mean_absolute_percentage_error(np.array(Y_test.target), Yhat_test)

            # Lists of errors for k tested dats for a tr training-set size
            train_mape_curve.append(mape_train)
            test_mape_curve.append(mape_test)

        # Remove 'inf' from the lists of loss functions
        train_mape_curve = [mape for mape in train_mape_curve if not (math.isinf(mape))]
        test_mape_curve = [mape for mape in test_mape_curve if not (math.isinf(mape))]

        mean_mape_train = sts.mean(train_mape_curve)
        mean_mape_test = sts.mean(test_mape_curve)
        train_mape_std = np.std(train_mape_curve)
        test_mape_std = np.std(test_mape_curve)
        train_mape_curve, test_mape_curve = [],[]

        tr_mape_std.append(train_mape_std)
        ts_mape_std.append(test_mape_std)
        tr_mape.append(mean_mape_train)
        ts_mape.append(mean_mape_test)

    train_mape_mean = np.asarray(tr_mape)
    train_mape_std = np.asarray(tr_mape_std)
    test_mape_mean = np.asarray(ts_mape)
    test_mape_std = np.asarray(ts_mape_std)
    

    fig = plt.figure()
    plt.rc('xtick',labelsize=28)
    plt.rc('ytick',labelsize=28)
    plt.grid()
    plt.fill_between(train_sizes, train_mape_mean - train_mape_std,
                        train_mape_mean + train_mape_std, alpha=0.1,
                        color='r')
    plt.fill_between(train_sizes, test_mape_mean - test_mape_std,
                        test_mape_mean + test_mape_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_mape_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mape_mean, 'o-', color='g', label='Test score')
    plt.xticks(train_sizes)
    plt.title( 'Learning Curves XGBoost' , fontsize=28)
    plt.xlabel('Training-set size [weeks]', fontsize=28)
    plt.ylabel('MAPE [%]', fontsize=28)
    plt.legend(loc='best', fontsize=28)
    fig.set_size_inches(28.5, 10.5)
    fig.savefig('learning-curves-xgboost.png')



    return
