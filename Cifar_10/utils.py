import os
import hashlib
import errno
import numpy as np
from numpy.testing import assert_array_almost_equal
from urllib import request


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    return md5c == md5


def download_url(url, root, filename, md5):
    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            request.urlretrieve(url, fpath)
        except Exception:
            if url.startswith('https'):
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead. Downloading ' + url + ' to ' + fpath)
                request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    root = os.path.expanduser(root)
    directories = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if prefix:
        directories = [os.path.join(root, d) for d in directories]
    return directories


def list_files(root, suffix, prefix=False):
    root = os.path.expanduser(root)
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)) and f.endswith(suffix)]
    if prefix:
        files = [os.path.join(root, f) for f in files]
    return files


def multiclass_noisify(y, P, random_state=0):
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0][0]

    return new_y


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print(f'Actual noise {actual_noise:.2f}')
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        for i in range(nb_classes):
            P[i, i] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print(f'Actual noise {actual_noise:.2f}')
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise


def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        return noisify_pairflip(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    elif noise_type == 'symmetric':
        return noisify_multiclass_symmetric(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")
