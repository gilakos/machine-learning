def create_dataset(data, seq_len=3, tt_split=0.90, normalise=True, pad=None):
    '''
    Convert an array of data into LSTM format test and train sequences
    :param data: array of data to convert
    :param seq_len: lookback value for number of entries per LSTM timestep: default 3
    :param tt_split: ratio of training data to test data: default = .90
    :param normalise_window: optional normalize
    :param pad: optional add padding of 0 to match dataset lengths
    :return: four arrays: x_train, y_train, x_test, y_test
    '''
    import numpy as np


    sequence_length = seq_len + 1
    if (pad):
        sequence_length = pad+1
        #print('pad active')
    result = []
    data_np = np.array(data)
    data_fl = data_np.astype(np.float)
    bounds = [np.amin(data_fl), np.amax(data_fl)]

    for index in range(len(data) - sequence_length):
        if (pad):
            x = []
            for i in range(0, pad-seq_len):
                x.append(data[index])
            for i in range(0, seq_len+1):
                x.append(data[index + i])
        else:
            x = data[index: index + sequence_length]
        result.append(x)

    if normalise:
        result = normalise_all(result)
    result = np.array(result)

    row = round(tt_split * result.shape[0])
    train = result[:int(row), :]
    # np.random.shuffle(train)
    if (pad):
        offset = seq_len
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, bounds


def normalise_windows(data):
    '''
    Normalize arrays of data in an array
    :param data: data to normalize
    :return: array of arrays of normalized data
    '''
    import numpy as np
    normalised_data = []
    for window in data:
        mm = [min(window), max(window)]  # minmax
        # print mm
        tr = [0.0, 1.0]  # target range
        normalised_window = []
        for p in window:
            p_scaled = (p - mm[0]) * (tr[1] - tr[0]) / (mm[1] - mm[0]) + tr[0]
            normalised_window.append(p_scaled)
        # normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


def normalise_all(data):
    '''
    Normalize all data in an array of arrays
    :param data: data to normalize
    :return: array of arrays of normalized data
    '''
    import numpy as np
    normalised_data = []
    temp = np.array(data)
    mm = [np.amin(temp), np.amax(temp)]  # minmax
    tr = [0.0, 1.0]  # target range
    for window in data:
        normalised_window = []
        for p in window:
            p_scaled = (p - mm[0]) * (tr[1] - tr[0]) / (mm[1] - mm[0]) + tr[0]
            normalised_window.append(p_scaled)
        # normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def scale_range(X, input_range=[0.0,1.0], target_range=[0.0,1.0]):
    '''
    Rescale a numpy array from input to target range
    :param X: data to scale
    :param input_range: optional input range for data: default 0.0:1.0
    :param target_range: optional target range for data: default 0.0:1.0
    :return: rescaled array, incoming range [min,max]
    '''
    import numpy as np
    orig_range = [np.amin(X), np.amax(X)]
    X_std = (X - input_range[0]) / (1.0*(input_range[1] - input_range[0]))
    X_scaled = X_std * (1.0*(target_range[1] - target_range[0])) + target_range[0]
    return X_scaled, orig_range

def predict_point_by_point(model, data):
    '''
    Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    :param model: model to predict
    :param data: data to input to model for prediction
    :return: predictions in numpy array
    '''
    import numpy as np
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequences(model, data, window_size=1, prediction_len=1):
    '''
    Predict sequence of X steps before shifting prediction run forward by X steps
    :param model: model to predict
    :param data: data to input to model for prediction
    :param window_size: size of window
    :param prediction_len: number of steps to predict
    :return: predictions in numpy array
    '''
    import numpy as np
    prediction_seqs = []
    for i in xrange(len(data)/prediction_len):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in xrange(prediction_len):
            predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs