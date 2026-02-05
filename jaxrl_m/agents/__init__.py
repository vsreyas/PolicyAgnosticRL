from .continuous.bc import BCAgent
from .continuous.calql import CalQLAgent
from .continuous.cql import ContinuousCQLAgent
from .continuous.ddpm_bc import DDPMBCAgent
from .continuous.iql import IQLAgent
from .continuous.parl_iql import PARLIQLAgent
from .continuous.parl_calql import PARLCalQLAgent
from .continuous.sac import SACAgent
from .continuous.diffusion_q_learning import DiffusionQLearningAgent
from .continuous.auto_regressive_transformer import AutoRegressiveTransformerAgent
from .continuous.pi_0 import PiPolicy
from .continuous.expo_pi import ExpoPiLearner
from .continuous.pi_vlm_cached.residual_td3 import PiResidualTD3Cache
from .continuous.pi_vlm_cached.residual_td3_grpo import PiResidualTD3GRPO

agents = {
    "ddpm_bc": DDPMBCAgent,
    "bc": BCAgent,
    "iql": IQLAgent,
    "parl_iql": PARLIQLAgent,
    "parl_calql": PARLCalQLAgent,
    "cql": ContinuousCQLAgent,
    "calql": CalQLAgent,
    "diffusion_q_learning": DiffusionQLearningAgent,
    "sac": SACAgent,
    "auto_regressive_transformer": AutoRegressiveTransformerAgent,
    "pi-0": PiPolicy,
    "expo": ExpoPiLearner,
    # Residual Agents working with VLM output of Pi0.5 #
    "pi_residual_td3": PiResidualTD3Cache,
    "pi_residual_td3_grpo": PiResidualTD3GRPO,
    ####################################################
}
