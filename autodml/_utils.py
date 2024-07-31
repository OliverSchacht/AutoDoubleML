import warnings

def assert_time(time, params_names):
    if not (isinstance(time, int) or time is None or isinstance(time, dict)):
            raise TypeError(f'time has to be of type int, dictionary or None. \
                              {type(time)} was provided.')
    if time is None:
        time = 60
        warnings.warn(f'No optimization time provided. Using default time. Time is equally split to all.')
    
    if isinstance(time, int):
        if time<0:
            raise ValueError(f'time has to be positive. \
                                {time} was provided')
        time = {key: int(time/len(params_names)) for key in params_names}
    
    else:
        if not all(key in time.keys() for key in params_names):
                raise ValueError(f'If time is a dictionary, it has to have keys {params_names}. \
                                {time.keys()} was provided')
        if not all(isinstance(time[key], int) for key in params_names):
                raise TypeError(f'If time is a dictionary, the values have to be of type int. \
                                {type(time)} was provided')
        if any(time[key]<0 for key in params_names):
                raise ValueError(f'If time is a dictionary, the values have to be positive. \
                                {time} was provided')
    return time
