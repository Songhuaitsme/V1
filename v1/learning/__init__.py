"""v1.0 variable-candidate learning and physical-time reward contracts."""

from .candidate_dqn import (
    CandidateDQNPolicy,
    CandidateDQNTrainer,
    CandidateDqnMetadata,
    CandidateSelectionTrace,
    SharedCandidateQNetwork,
    double_dqn_target,
    mask_q_values,
    select_candidate_id,
    validate_checkpoint_metadata,
)
from .features import (
    CANDIDATE_FEATURE_NAMES,
    CandidateFeatureConfig,
    CandidateFeatureEncoder,
)
from .replay import CandidateReplayBuffer, ReplayTransition, TimestampedReward
from .reward import DecisionRecord, EventRewardBuffer, GammaClock, RewardAssembler

__all__ = [
    "CANDIDATE_FEATURE_NAMES",
    "CandidateDQNPolicy",
    "CandidateDQNTrainer",
    "CandidateDqnMetadata",
    "CandidateSelectionTrace",
    "CandidateFeatureConfig",
    "CandidateFeatureEncoder",
    "CandidateReplayBuffer",
    "DecisionRecord",
    "EventRewardBuffer",
    "GammaClock",
    "ReplayTransition",
    "RewardAssembler",
    "SharedCandidateQNetwork",
    "TimestampedReward",
    "double_dqn_target",
    "mask_q_values",
    "select_candidate_id",
    "validate_checkpoint_metadata",
]
