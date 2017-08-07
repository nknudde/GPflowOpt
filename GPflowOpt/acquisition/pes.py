# Copyright 2017 Joachim van der Herten
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .acquisition import Acquisition

from GPflow.model import Model
from GPflow.param import DataHolder, AutoFlow, ParamList, Param
from GPflow import settings

import numpy as np
import tensorflow as tf

stability = settings.numerics.jitter_level
float_type = settings.dtypes.float_type


class PredictiveEntropySearch(Acquisition):
    """
    Predictive entropy search acquisition function for single-objective global optimization.
    Introduced by (Lobato et al., 2014).

    Key reference:

    ::

       @inproceedings{hernandez2014predictive,
          title={Predictive entropy search for efficient global optimization of black-box functions},
          author={Hern{\'a}ndez-Lobato, Jos{\'e} Miguel and Hoffman, Matthew W and Ghahramani, Zoubin},
          booktitle={Advances in neural information processing systems},
          pages={918--926},
          year={2014}
        }
    """

    def __init__(self, model):
        super(PredictiveEntropySearch, self).__init__(model)
        assert (isinstance(model, Model))
        self.setup()

    def setup(self):
        super(PredictiveEntropySearch, self).setup()
        Xstar = 0  # TODO
        X = self.models[0].wrapped.X.value
        N = X.shape[0]
        fmean, fvar = self.models[0].predict_f(np.stack((Xstar, X), axis=0))
        A = np.zeros((N, 2, 2))
        b = np.zeros((N, 2))

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]), (float_type, [None, 2, 2]),
              (float_type, [None, 2]))
    def ep_iteration(self, fmean, fvar, A, b):
        L = tf.cholesky(fvar)
        N = tf.shape(fvar)[0] - 1

        Vt_diag = tf.diag(tf.stack(
            [tf.reduce_sum(tf.slice(A, [0, 1, 1], [-1, 1, 1])), tf.squeeze(tf.slice(A, [0, 0, 0], [-1, 1, 1]))],
            axis=0))
        Vt_rest = tf.pad(tf.expand_dims(tf.squeeze(tf.slice(A, [0, 0, 1], [-1, 1, 1])), axis=1), [[1, 0], [0, N]],
                         mode="constant")
        Vt = Vt_diag + tf.transpose(Vt_rest) + Vt_rest
        mt = tf.stack([tf.reduce_sum(tf.slice(b, [0, 1], [-1, 1])), tf.slice(b, [0, 0], [-1, 1])], axis=0)

        tmp1 = tf.eye(N + 1) + tf.matmul(tf.transpose(L), tf.matmul(Vt, L))
        tmp1L = tf.cholesky(tmp1)
        tmp2 = tf.matrix_triangular_solve(tmp1L, tf.transpose(L))
        Vf = tf.matmul(tf.transpose(tmp2), tmp2)
        mf = fmean - tf.matmul(Vf, tf.matmul(Vt, fmean) + mt)

        Vfn0_diag = tf.matrix_diag(tf.stack(
            [tf.reshape(tf.slice(Vf, [1, 1], -1, -1), [N, 1]), tf.tile(tf.slice(Vf, [0, 0], [1, 1]), [N, 1])], axis=1))
        Vfn0_rest = tf.pad(tf.expand_dims(tf.slice(Vf, [1, 0], [-1, 1]), 2), [[0, 0], [1, 0], [0, 1]])
        Vfn0 = Vfn0_diag + Vfn0_rest + tf.matrix_transpose(Vfn0_rest)
        Vfn0L = tf.cholesky(Vfn0)
        mfn0 = tf.expand_dims(
            tf.stack([tf.slice(mf, [1, 0], [-1, -1]), tf.tile(tf.slice(mf, [0, 0], [1, -1]), [N, 1])], axis=1), 2)

        tmp3 = tf.tile(tf.expand_dims(tf.eye(2), 0), [N, 1, 1]) - tf.matmul(tf.matrix_transpose(Vfn0L),
                                                                            tf.matmul(A, Vfn0L))
        tmp3L = tf.cholesky(tmp3)
        tmp4 = tf.matrix_triangular_solve(tmp3L, tf.matrix_transpose(Vfn0L))

        Vn = tf.matmul(tf.matrix_transpose(tmp4), tmp4)
        mn = mfn0 + tf.matmul(Vn, tf.matmul(A, mfn0) - tf.expand_dims(b, 2))

        alpha = tf.squeeze(tf.slice(mn, [0, 0, 0], [-1, 1, 1]) - tf.slice(mn, [0, 1, 0], [-1, 1, 1])) / tf.sqrt(
            tf.reduce_sum(2 * tf.matrix_diag_part(Vn), axis=1) - tf.reduce_sum(Vn, [1, 2]))

        norm = tf.contrib.distributions.Normal()
        logZ = norm.log_cdf(alpha)

        dlogZdm = tf.gradients(logZ, mn)
        dlogZdV = tf.gradients(logZ, Vn)

        Vfn0_new = Vn - tf.matmul(Vn, tf.matmul(tf.matmul(dlogZdm, tf.matrix_transpose(dlogZdm))-2*dlogZdV, Vn))
        mfn0_new = mn + tf.matmul(Vn, dlogZdm)

        Vfn0_new_chol = tf.cholesky(Vfn0_new)
        VnL = tf.cholesky(Vn)

        tmp5 = tf.matrix_triangular_solve(Vfn0_new_chol, VnL)
        tmp6 = tf.cholesky(tf.matmul(tf.matrix_transpose(tmp5), tmp5) + tf.expand_dims(tf.eye(2),0))
        tmp7 = tf.matrix_triangular_solve(tf.matrix_transpose(VnL), tmp6, lower=False)
        An = tf.matmul(tmp7, tf.matrix_transpose(tmp7))
        bn = tf.squeeze(tf.cholesky_solve(Vfn0_new_chol, mfn0_new) - tf.cholesky_solve(VnL, mfn0_new))

        return An, bn

    def build_acquisition(self, Xcand):
        return None
