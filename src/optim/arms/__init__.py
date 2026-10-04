from src.optim.arms.baselines_arms import BestOfB, ParaphraseSearch, RandomSearch
from src.optim.arms.evolutionary import DE, GA, PromptBreeder
from src.optim.arms.qd import NSGA2, MAPElites
from src.optim.arms.reflective import GEPA, OPRO, ProTeGi
from src.optim.arms.gepa_official import OfficialGEPA

ARMS = {a.name: a for a in (RandomSearch, ParaphraseSearch, BestOfB, GA, DE, PromptBreeder, OPRO, ProTeGi, GEPA,
                            MAPElites, NSGA2, OfficialGEPA)}
