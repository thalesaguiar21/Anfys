from numpy import array


line_size = 100
separator = '-'


def __line_sep():
    return separator * line_size + '\n'


def layer_output_msg(output, layer_num):
    """ Build a formatted string for a layer's output

    Params
    ------
    output : list of doubles
        The values to be displayed
    layer_num : int
        The layer's number

    Returns
    -------
    msg : string
        A formatted ouput string for the given layer
    """
    msg = 'Layer {} output\n'.format(layer_num)
    msg += __line_sep()
    msg += str(array(output)) + '\n'
    msg += __line_sep()
    return msg


def begin_training_msg(feature, expected):
    """ Build a formatted string to be displayed when the training starts

    Params
    ------
    feature : list of double
        The feature vector
    expected : object
        The expected output for the given features

    Returns
    -------
    msg : string
        A formatted string to print the initial and targeted data for training
    """
    msg = '\nTraining for:'
    msg += '\n' + __line_sep() + __line_sep()
    msg += 'INPUT:\n' + str(array(feature)) + '\n'
    msg += 'EXPECTED:\n' + str(array(expected))
    msg += '\n' + __line_sep() + __line_sep()
    return msg


def epoch_error_msg(epoch, error):
    """ Build a formatted string for printing the error at each epoch

    Params
    ------
    epoch : int
        The current epoch
    error : double
        The total error at the given epoch

    Returns
    -------
    msg : string
        The formatted string for printing the error at each epoch
    """
    msg = 'Error for {}-th epoch is {}.\n'.format(epoch, error)
    return msg


def convergence_msg(epoch):
    return 'Convergence occurred at epoch ' + str(epoch) + '\n'


def end_epoch_msg(epoch, pred, error):
    return 'Final prediction was {} with {} of error!'.format(
        array(pred), error
    )
