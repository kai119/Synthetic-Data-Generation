import pandas as pd
from sklearn.preprocessing import StandardScaler


def transform_gravity_data(df, filename):
    """Transform gravity data into data that is ready to be fed into a NN.

    Removes Speed On Impact (m/s), Deceleration, and Force columns,
    One-hot encodes the Type column,
    Transforms the target variable into numerical values
    Scales the Height (m), and Mass (KG) columns using StandardScaler.

    Parameters
    ----------
    df: pd.Dataframe
        the gravity data to transform
    filename
        the file name to send the transformed data to

    Returns
    -------
    pd.Dataframe
        the transformed gravity data
    """

    if {'Height (m)', 'Mass (KG)', 'Type', 'OUTCOME'}.issubset(df.columns):

        if {'Speed On Impact (m/s)', 'Deceleration', 'Force'}.issubset(df.columns):
            # Drop columns not being used
            df.drop(['Speed On Impact (m/s)', 'Deceleration', 'Force'], inplace=True, axis=1)

        # One-hot encode the Type Column
        types = pd.get_dummies(df['Type'])
        df.drop('Type', inplace=True, axis=1)

        # Transform target variable into numerical values
        outcomes = {
            'UNBROKEN': 0,
            'BROKEN': 1
        }
        df['OUTCOME'] = df['OUTCOME'].map(outcomes)

        # Split X and y and scale the X variables
        X = df.drop(['OUTCOME'], axis=1)
        y = df['OUTCOME']

        scaler = StandardScaler()
        X_transformed = scaler.fit_transform(X)
        X_transformed_df = pd.DataFrame(X_transformed, columns=X.columns)

        # Re-attach y and type column to X and save df to a file
        X_transformed_df['OUTCOME'] = y
        X_transformed_df = X_transformed_df.join(types)
        X_transformed_df.to_excel('data/' + filename, index=False)

        # Return the transformed df
        return X_transformed_df
    else:
        print(f'ERROR: df does not contain correct columns. Columns in df: [{df.columns}]')
        return df
