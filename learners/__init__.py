from .q_learner import QLearner
from .iqn_learner import IQNLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .rest_q_clean import RestQLearner
from .dresq_nocentral_learner import NoCentralDResQLearner

from .dresq_new import DResQ
from .resz_learner import ResZ
from .resz_v1_learner import ResZ_todo
# from .rest_q_learner_central import RestQLearnerCentral

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["iqn_learner"] = IQNLearner
REGISTRY["restq_learner_clean"] = RestQLearner
REGISTRY['dresq_nocentral_learner'] = NoCentralDResQLearner
REGISTRY['dresq_new'] = DResQ
REGISTRY['resz_learner'] = ResZ
REGISTRY['resz_simple'] = ResZ_todo