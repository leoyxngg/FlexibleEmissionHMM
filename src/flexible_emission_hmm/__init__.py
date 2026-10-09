from .emissions import EmissionModel, GaussianEmission, GaussianMixtureEmission
from .hmm import HMM
from .inference import ExactInference, InferenceEngine
from .parameters import EvaluationResult, InferenceResult, StateParameters
from .training import BaumWelchTrainer, Trainer

__all__ = [
    "HMM",
    "EmissionModel",
    "GaussianEmission",
    "GaussianMixtureEmission",
    "InferenceEngine",
    "ExactInference",
    "Trainer",
    "BaumWelchTrainer",
    "StateParameters",
    "InferenceResult",
    "EvaluationResult",
]


def main() -> None:
    print("Hello from flexibleemissionhmm!")
