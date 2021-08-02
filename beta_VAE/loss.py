import tensorflow as tf
import numpy as np
from tensorflow.linalg import matvec

def rota_cross_loss(model, x, d, r_x):
    c, s = np.cos(d), np.sin(d)
    latent = model.latent_dim
    r_m = np.identity(latent)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, s], [-s, c]
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    phi_z = rotate_vector(z, r_m)
    phi_x = model.decode(phi_z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=r_x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    return -tf.reduce_mean(logx_z)



def rotate_vector(vector, matrix):
    matrix = tf.cast(matrix, tf.float32)
    test = matvec(matrix, vector)
    return test

def ori_cross_loss(model, x, d, r_x):
    mean, logvar = model.encode(r_x)
    r_z = model.reparameterize(mean, logvar)
    c, s = np.cos(d), np.sin(d)
    latent = model.latent_dim
    r_m = np.identity(latent)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, -s], [s, c]
    phi_z = rotate_vector(r_z, r_m)
    phi_x = model.decode(phi_z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    return -tf.reduce_mean(logx_z)


def reconstruction_loss(model, X, y):
    mean, logvar = model.encode(X)
    Z = model.reparameterize(mean, logvar)
    X_pred = model.decode(Z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=X_pred, labels=X)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    h = model.projection(Z)
    encode_loss = top_loss(model, h, y)
    return -tf.reduce_mean(logx_z) + encode_loss, h


def kl_divergence(mean, logvar):
    summand = tf.math.square(mean) + tf.math.exp(logvar) - logvar  - 1
    return (0.5 * tf.reduce_sum(summand, [1]))

def gaussian_log_density(samples, mean, logvar):
    pi = tf.constant(np.pi)
    normalization = tf.math.log(2. * pi)
    inv_sigma = tf.math.exp(-logvar)
    tmp = (samples - mean)
    return -0.5 * (tmp * tmp * inv_sigma + logvar + normalization)

def estimate_entropies(qz_samples, mean, logvar):
    log_q_z_prob = gaussian_log_density(
        tf.expand_dims(qz_samples,1),  tf.expand_dims(mean,0),
    tf.expand_dims(logvar, 0))

    log_q_z_product = tf.math.reduce_sum(
        tf.math.reduce_logsumexp(log_q_z_prob, axis=1, keepdims=False),
        axis=1, keepdims=False
    )

    log_qz = tf.math.reduce_logsumexp(
        tf.math.reduce_sum(log_q_z_prob, axis=2, keepdims=False)
    )
    return log_qz, log_q_z_product

def rotate_vector(vector, matrix):
    matrix = tf.cast(matrix, tf.float32)
    test = matvec(matrix, vector)
    return test

def compute_loss(model, classifier, x, y, gamma=1, loss='cross_entropy'):
    beta = model.beta
    mean, logvar = model.encode(x)
    features = model.reparameterize(mean, logvar)
    identity = tf.expand_dims(tf.cast(y, tf.float32), 1)
    z = tf.concat([features, identity], axis=1)
    x_logit = model.decode(z)
    '''
    reco_loss = reconstruction_loss(x_logit, x)
    kl_loss = kl_divergence(logvar, mean)
    beta_loss = reco_loss + kl_loss * beta
    '''
    kl_loss = tf.reduce_mean(kl_divergence(mean, logvar))
    h = classifier.projection(x)
    classifier_loss = top_loss(classifier, h, y)
    if (loss=='cross_entropy'):
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logx_z = tf.reduce_mean(tf.reduce_sum(cross_ent, axis=[1, 2, 3]))
        log_qz, logq_z_product = estimate_entropies(features, mean, logvar)
        tc = tf.reduce_mean(log_qz - logq_z_product)
        return tf.reduce_mean(logx_z + kl_loss  + (beta - 1) * tc), h, classifier_loss
    elif(loss == 'mean_square_loss'):
        logx_z = tf.reduce_mean(tf.losses.mean_squared_error(y_true=x, y_pred=x_logit))
        return tf.reduce_mean(logx_z + beta * kl_loss), h, classifier_loss

def dual_loss(model, classifier, x, y, gamma=1):
    beta = model.beta
    mean, logvar = model.encode(x)
    features = model.reparameterize(mean, logvar)
    identity = tf.expand_dims(tf.cast(y, tf.float32), 1)
    z = tf.concat([features, identity], axis=1)
    rotation, position = model.decode(z)
    kl_loss = tf.reduce_mean(kl_divergence(mean, logvar))
    h = classifier.projection(x)
    classifier_loss = top_loss(classifier, h, y)
    x_h = classifier.projection(tf.concat([rotation, position], axis=2))
    gen_loss = top_loss(classifier, x_h, y) * gamma
    log_qz, logq_z_product = estimate_entropies(features, mean, logvar)
    tc = tf.reduce_mean(log_qz - logq_z_product)

    rotation_loss = tf.reduce_mean(tf.losses.mean_squared_error(y_true=x[:, :, :3, :], y_pred=rotation))
    position_loss = tf.reduce_mean(tf.losses.mean_squared_error(y_true=x[:, :, 3:, :], y_pred=position))


    return tf.reduce_mean(rotation_loss + beta * kl_loss + gen_loss), \
           tf.reduce_mean(position_loss +  beta * kl_loss + gen_loss),\
           h, classifier_loss


def com_clr_loss(model, x, g, y, gamma=1):
    beta = model.beta
    mean, logvar = model.encode(x)
    z, features = model.reparameterize(mean, logvar, g)
    rotation, position = model.decode(z)
    kl_loss = tf.reduce_mean(kl_divergence(mean, logvar))
    h = model.projection(z)
    classifier_loss = top_loss(model, h, y)
    log_qz, logq_z_product = estimate_entropies(features, mean, logvar)
    tc = tf.reduce_mean(log_qz - logq_z_product)

    rotation_loss = tf.reduce_mean(tf.losses.mean_squared_error(y_true=x[:, :, :3, :], y_pred=rotation))
    position_loss = tf.reduce_mean(tf.losses.mean_squared_error(y_true=x[:, :, 3:, :], y_pred=position))


    return tf.reduce_mean(position_loss + rotation_loss + beta * kl_loss + classifier_loss), h


def top_loss(model, h, y):
    classes = model.num_cls
    labels = tf.one_hot(y, classes)
    loss_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=h
    ))

    return loss_t


def negative_entropy(data, normalize=False, max_value=None):
    softmax = tf.nn.softmax(data, 1)
    log_softmax = tf.nn.log_softmax(data, 1)
    entropy = softmax * log_softmax
    entropy = tf.math.reduce_sum(-1.0 * entropy, 1)
    # normalize [0 ~ 1]
    if normalize:
        normalized_entropy = entropy / max_value
        return -normalized_entropy

    return entropy

def confidence_function(model, data, target='softmax'):
    result = model.projection(data)
    if (target=='softmax'):
        conf = tf.math.reduce_max(tf.nn.softmax(result), 1)
    elif (target == 'negative_entropy'):
        value_for_normalizing = 2.302585
        conf= negative_entropy(result,
                                normalize=True,
                                max_value=value_for_normalizing)
    elif (target == 'margin'):
        conf, _ = tf.math.top_k(tf.nn.softmax(result), k=2)
        conf = conf[:, 0] - conf[:, 1]
    return conf, result.numpy().argmax(-1)

def true_positive(y_pred, y_true):
    return np.sum(y_pred[y_pred == y_true]==1)

def false_negative(y_pred, y_true):
    return np.sum(y_pred[y_pred!=y_true]==0)

def true_negative(y_pred, y_true):
    return np.sum(y_pred[y_pred == y_true]==0)

def false_positive(y_pred, y_true):
    return np.sum(y_pred[y_pred!=y_true]==1)

def sensitivity(y_pred, y_true):
    t_p =  true_positive(y_pred, y_true)
    f_n = false_negative(y_pred, y_true)
    if (t_p == 0):
        if (f_n + t_p == 0):
            return 1
        else:
            return 0
    else:
        return t_p/(t_p + f_n)

def specifity(y_pred, y_true):
    t_n = true_negative(y_pred, y_true)
    f_p = false_positive(y_pred, y_true)
    if (t_n == 0):
        if (t_n + f_p == 0):
            return 1
        else:
            return 0
    else:
        return t_n/(t_n + f_p)

def g_means(y_pred, y_true):
    return np.sqrt(sensitivity(y_pred, y_true) * specifity(y_pred, y_true))

def get_gMeans(y_pred, y_true):
    g_means_set = []
    c = np.bincount(y_true.flatten())
    for i in range(len(c)):
        tmp_pred = np.array([1 if label==i else 0 for label in y_pred])
        tmp_true = np.array([1 if label==i else 0 for label in y_true])
        g_means_set.append(g_means(tmp_pred, tmp_true))
    return np.mean(g_means_set)

def acsa(y_pred, y_true):
    correct = np.sum(y_pred[y_pred == y_true]==1)
    pred = np.sum(y_pred==1)
    if (correct == 0):
        if (pred == 0):
            return 1
        else:
            return 0
    else:
        return correct/pred

def acc_metrix(y_pred, y_true):
    g_means_set = []
    acsa_set = []
    c = np.bincount(y_true.flatten())
    for i in range(len(c)):
        tmp_pred = np.array([1 if label==i else 0 for label in y_pred])
        tmp_true = np.array([1 if label==i else 0 for label in y_true])
        g_means_set.append(g_means(tmp_pred, tmp_true))
        acsa_set.append(acsa(tmp_pred, tmp_true))
    return np.mean(g_means_set), np.mean(acsa_set)