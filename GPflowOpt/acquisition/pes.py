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

    def build_acquisition(self, Xcand):
        return None