"""V2 training entry point with idle-cycle skipping enabled by default."""

from v1.train_v1 import main as _training_main


def main():
    _training_main(
        system_version="2.0",
        default_output="artifacts/v2/logs/candidate_dqn.pt",
        default_candidate_chunk_size=65536,
        default_checkpoint_every=50000,
    )


if __name__ == "__main__":
    main()
