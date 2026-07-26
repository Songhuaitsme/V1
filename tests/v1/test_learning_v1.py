from dataclasses import replace
import random
import unittest

import networkx as nx
import numpy as np
import torch

from v1.domain.models import TaskSpec
from v1.domain.units import TimeConverter
from v1.learning import (
    CandidateDQNPolicy,
    CandidateDQNTrainer,
    CandidateFeatureConfig,
    CandidateFeatureEncoder,
    CandidateReplayBuffer,
    DecisionRecord,
    GammaClock,
    ReplayTransition,
    RewardAssembler,
    SharedCandidateQNetwork,
    TimestampedReward,
    double_dqn_target,
    select_candidate_id,
)
from v1.scheduler import (
    CandidateGenerator,
    ReservationCalendar,
    StaticPathProvider,
    TransmissionModel,
)


class CandidateDqnV1Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.network = SharedCandidateQNetwork(4, 6, hidden_dim=8)

    # DQN-001
    def test_variable_candidate_count_controls_output_length(self):
        state = torch.zeros(4)
        for count in (1, 3, 17):
            with self.subTest(count=count):
                output = self.network(state, torch.zeros((count, 6)))
                self.assertEqual(tuple(output.shape), (1, count))

    # DQN-002
    def test_candidate_scoring_is_permutation_equivariant(self):
        state = torch.tensor([1.0, 2.0, 3.0, 4.0])
        candidates = torch.arange(18, dtype=torch.float32).reshape(3, 6)
        original = self.network(state, candidates).detach().numpy()[0]
        permutation = [2, 0, 1]
        permuted = self.network(state, candidates[permutation]).detach().numpy()[0]
        np.testing.assert_allclose(permuted, original[permutation], rtol=0, atol=1e-7)

    # DQN-003 / DQN-004
    def test_ties_and_masks_are_deterministic(self):
        self.assertEqual(
            select_candidate_id(("z", "a"), (1.0, 1.0)),
            "a",
        )
        self.assertEqual(
            select_candidate_id(("best-invalid", "valid"), (100.0, 1.0), (False, True)),
            "valid",
        )
        with self.assertRaises(ValueError):
            select_candidate_id(("x",), (1.0,), (False,))

    # DQN-005 / DQN-006
    def test_double_dqn_uses_only_next_feasible_variable_set(self):
        target = double_dqn_target(
            2.0,
            False,
            0.5,
            online_next_q=(1.0, 9.0, 3.0),
            target_next_q=(10.0, 20.0, 30.0),
            valid_mask=(True, False, True),
        )
        self.assertEqual(target, 17.0)
        self.assertEqual(
            double_dqn_target(2.0, False, 0.5, (), ()),
            2.0,
        )
        self.assertEqual(
            double_dqn_target(2.0, True, 0.5, (100.0,), (100.0,)),
            2.0,
        )

    def _transition(self):
        return ReplayTransition(
            global_state_before=(1.0, 2.0),
            selected_candidate_id="candidate-a",
            selected_candidate_features=(3.0, 4.0),
            reward=5.0,
            global_state_after=(6.0, 7.0),
            next_candidate_features=((8.0, 9.0), (10.0, 11.0)),
            decision_time_sim=1.0,
            next_transition_time_sim=2.0,
            elapsed_seconds=10.0,
            gamma_elapsed=0.9,
            terminal=False,
            timestamped_event_rewards=(TimestampedReward(1.5, -1.0, "d", "EXPIRED"),),
        )

    # DQN-007 / DQN-012
    def test_replay_round_trip_and_append_only_history(self):
        transition = self._transition()
        restored = ReplayTransition.from_json(transition.to_json())
        self.assertEqual(restored, transition)
        buffer = CandidateReplayBuffer(2)
        buffer.add(transition)
        assembler = RewardAssembler(GammaClock(0.99, TimeConverter(1.0)))
        assembler.buffer_event(TimestampedReward(3.0, 100.0, "later", "COMPLETED"))
        self.assertEqual(buffer.items()[0], transition)

    def test_replay_random_batch_and_checkpoint_state(self):
        first = self._transition()
        second = replace(first, selected_candidate_id="candidate-b", reward=6.0)
        buffer = CandidateReplayBuffer(3)
        buffer.add(first)
        buffer.add(second)
        sample_a = buffer.sample(2, random.Random(11))
        sample_b = buffer.sample(2, random.Random(11))
        self.assertEqual(sample_a, sample_b)
        restored = CandidateReplayBuffer(3)
        restored.load_state_dict(buffer.state_dict())
        self.assertEqual(restored.items(), buffer.items())

    def test_context_backed_batch_regenerates_next_candidates_in_chunks(self):
        calls = []

        def provider(context):
            calls.append(context)
            yield ((8.0, 9.0),)
            yield ((10.0, 11.0),)

        online = SharedCandidateQNetwork(2, 2, 4)
        target = SharedCandidateQNetwork(2, 2, 4)
        trainer = CandidateDQNTrainer(
            online,
            target,
            1e-3,
            candidate_chunk_size=1,
            next_candidate_provider=provider,
        )
        trainer.update_target()
        transition = replace(
            self._transition(),
            next_candidate_features=(),
            next_candidate_context="snapshot-context",
        )
        loss = trainer.train_batch((transition, transition))
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(calls, ["snapshot-context", "snapshot-context"])

    # DQN-008 plus actual policy return contract
    def test_candidate_policy_has_no_wait_or_reject_action(self):
        self.assertFalse(hasattr(CandidateDQNPolicy, "WAIT_ACTION"))
        self.assertFalse(hasattr(CandidateDQNPolicy, "REJECT_ACTION"))

    def test_complete_exploration_materializes_only_selected_and_earliest(self):
        graph = nx.Graph()
        graph.add_node("L")
        calendar = ReservationCalendar({"L": 2.0}, {})
        generator = CandidateGenerator(
            ("L",),
            1.0,
            StaticPathProvider(graph, max_paths_per_target=1),
            TransmissionModel(TimeConverter(1.0), 200000.0),
            calendar,
        )
        task = TaskSpec.create(
            task_id="explore",
            arrival_time_sim=0.0,
            source_node="L",
            cpu_demand=1.0,
            execution_duration_sim=1.0,
            data_size_mb=0.0,
            bandwidth_demand_mbps=1.0,
            sla_type="Hard",
            latest_start_limit_sim=4.0,
        )
        optimized_metric_starts = []
        legacy_metric_starts = []

        def evaluator(calls):
            def evaluate(**kwargs):
                calls.append(kwargs["compute_start_sim"])
                return {
                    "system_cost_yuan": kwargs["compute_start_sim"] + 1.0,
                    "green_coverage": 0.5,
                    "marginal_green_energy_mwh": 0.1,
                    "green_absorption_delta": 0.2,
                    "green_opportunity": True,
                }

            return evaluate

        optimized_stream = generator.prepare_complete_stream(
            task,
            0.0,
            metric_evaluator=evaluator(optimized_metric_starts),
        )
        legacy_stream = generator.prepare_complete_stream(
            task,
            0.0,
            metric_evaluator=evaluator(legacy_metric_starts),
        )
        encoder = CandidateFeatureEncoder(
            {"L": 0},
            CandidateFeatureConfig(10.0, 10.0, 1.0, 2.0, 10.0),
        )
        optimized_network = SharedCandidateQNetwork(4, encoder.feature_dim, 8)
        legacy_network = SharedCandidateQNetwork(4, encoder.feature_dim, 8)
        legacy_network.load_state_dict(optimized_network.state_dict())
        state_provider = lambda task: np.zeros(4, dtype=np.float32)
        optimized = CandidateDQNPolicy(
            optimized_network,
            encoder,
            state_provider,
            epsilon=1.0,
            random_seed=23,
            record_selection_traces=True,
            candidate_chunk_size=2,
        )
        legacy = CandidateDQNPolicy(
            legacy_network,
            encoder,
            state_provider,
            epsilon=1.0,
            random_seed=23,
            record_selection_traces=True,
            candidate_chunk_size=2,
        )

        fast_selection = optimized.select_complete_stream(
            optimized_stream, task=task
        )
        reference_selection = legacy.select_stream(
            legacy_stream.iter_candidates(),
            task=task,
            context=legacy_stream.context,
        )

        self.assertEqual(
            fast_selection.selected_candidate,
            reference_selection.selected_candidate,
        )
        self.assertEqual(
            fast_selection.earliest_candidate,
            reference_selection.earliest_candidate,
        )
        self.assertEqual(
            fast_selection.candidate_count,
            reference_selection.candidate_count,
        )
        self.assertEqual(
            fast_selection.candidate_set_hash,
            reference_selection.candidate_set_hash,
        )
        self.assertLessEqual(len(optimized_metric_starts), 2)
        self.assertEqual(
            len(legacy_metric_starts),
            reference_selection.candidate_count,
        )
        trace = optimized.pop_selection_traces()[0]
        self.assertIsNone(trace.q_min)
        self.assertIsNone(trace.q_max)
        self.assertIsNone(trace.q_mean)

        training_metric_starts = []
        training_stream = generator.prepare_complete_stream(
            task,
            0.0,
            metric_evaluator=evaluator(training_metric_starts),
        )
        training_network = SharedCandidateQNetwork(
            4, encoder.feature_dim, 8
        )
        training_network.load_state_dict(
            optimized_network.state_dict()
        )
        training = CandidateDQNPolicy(
            training_network,
            encoder,
            state_provider,
            epsilon=1.0,
            random_seed=23,
            candidate_chunk_size=2,
        )
        training.audit_candidate_set_hash = False
        training_selection = training.select_complete_stream(
            training_stream, task=task
        )
        self.assertEqual(
            training_selection.selected_candidate,
            reference_selection.selected_candidate,
        )
        self.assertEqual(
            training_selection.earliest_candidate,
            reference_selection.earliest_candidate,
        )
        self.assertEqual(
            training_selection.candidate_count,
            reference_selection.candidate_count,
        )
        self.assertEqual(
            training.random.random(),
            optimized.random.random(),
        )
        self.assertLessEqual(len(training_metric_starts), 2)

    def test_direct_record_features_equal_candidate_object_features(self):
        graph = nx.Graph()
        graph.add_node("L")
        calendar = ReservationCalendar({"L": 2.0}, {})
        generator = CandidateGenerator(
            ("L",),
            1.0,
            StaticPathProvider(graph, max_paths_per_target=1),
            TransmissionModel(TimeConverter(1.0), 200000.0),
            calendar,
        )
        task = TaskSpec.create(
            task_id="direct-features",
            arrival_time_sim=0.0,
            source_node="L",
            cpu_demand=1.0,
            execution_duration_sim=1.0,
            data_size_mb=0.0,
            bandwidth_demand_mbps=1.0,
            sla_type="Hard",
            latest_start_limit_sim=4.0,
        )

        def metrics(**kwargs):
            start = kwargs["compute_start_sim"]
            return {
                "system_cost_yuan": start + 1.25,
                "green_coverage": 0.4 + start * 0.01,
                "marginal_green_energy_mwh": 0.2,
                "green_absorption_delta": start * 0.02,
                "green_opportunity": start > 0.0,
            }

        stream = generator.prepare_complete_stream(
            task, 0.0, metric_evaluator=metrics
        )
        encoder = CandidateFeatureEncoder(
            {"L": 0},
            CandidateFeatureConfig(10.0, 10.0, 1.0, 2.0, 10.0),
        )
        object_features = tuple(
            encoder.encode(item, stream.earliest_compute_start_sim)
            for item in stream.iter_candidates()
        )
        direct_features = tuple(
            feature
            for chunk in generator.feature_chunks_from_context(
                stream.context,
                encoder,
                metric_evaluator=metrics,
                chunk_size=2,
            )
            for feature in chunk
        )

        self.assertEqual(direct_features, object_features)

    def test_batched_replay_features_are_float32_identical_to_scalar(self):
        graph = nx.Graph()
        graph.add_node("L")
        calendar = ReservationCalendar({"L": 2.0}, {})
        generator = CandidateGenerator(
            ("L",),
            1.0,
            StaticPathProvider(graph, max_paths_per_target=1),
            TransmissionModel(TimeConverter(1.0), 200000.0),
            calendar,
        )
        task = TaskSpec.create(
            task_id="batch-features",
            arrival_time_sim=0.0,
            source_node="L",
            cpu_demand=1.0,
            execution_duration_sim=1.0,
            data_size_mb=0.0,
            bandwidth_demand_mbps=1.0,
            sla_type="Soft",
            preferred_start_limit_sim=4.0,
            latest_start_limit_sim=4.8,
        )

        def metrics(**kwargs):
            start = kwargs["compute_start_sim"]
            return {
                "system_cost_yuan": start + 1.25,
                "green_coverage": 0.4 + start * 0.01,
                "marginal_green_energy_mwh": 0.2,
                "green_absorption_delta": start * 0.02,
                "green_opportunity": start > 0.0,
            }

        def metrics_batch(**kwargs):
            starts = np.asarray(kwargs["compute_start_sim"])
            return {
                "system_cost_yuan": starts + 1.25,
                "green_coverage": 0.4 + starts * 0.01,
                "marginal_green_energy_mwh": np.full(
                    starts.shape, 0.2
                ),
                "green_absorption_delta": starts * 0.02,
                "green_opportunity": starts > 0.0,
            }

        metrics.evaluate_batch = metrics_batch
        stream = generator.prepare_complete_stream(
            task, 0.0, metric_evaluator=metrics
        )
        encoder = CandidateFeatureEncoder(
            {"L": 0},
            CandidateFeatureConfig(10.0, 10.0, 1.0, 2.0, 10.0),
        )
        scalar = np.asarray(
            [
                encoder.encode(
                    candidate, stream.earliest_compute_start_sim
                )
                for candidate in stream.iter_candidates()
            ],
            dtype=np.float32,
        )
        batched = np.concatenate(
            tuple(
                generator.feature_chunks_from_context(
                    stream.context,
                    encoder,
                    metric_evaluator=metrics,
                    chunk_size=2,
                )
            ),
            axis=0,
        )
        np.testing.assert_array_equal(batched, scalar)

        fast_network = SharedCandidateQNetwork(
            4, encoder.feature_dim, 8
        )
        reference_network = SharedCandidateQNetwork(
            4, encoder.feature_dim, 8
        )
        reference_network.load_state_dict(
            fast_network.state_dict()
        )
        state_provider = lambda selected_task: np.zeros(
            4, dtype=np.float32
        )
        fast_policy = CandidateDQNPolicy(
            fast_network,
            encoder,
            state_provider,
            epsilon=0.0,
            random_seed=41,
            record_selection_traces=True,
            candidate_chunk_size=2,
        )
        fast_policy.audit_candidate_set_hash = False
        reference_policy = CandidateDQNPolicy(
            reference_network,
            encoder,
            state_provider,
            epsilon=0.0,
            random_seed=41,
            record_selection_traces=True,
            candidate_chunk_size=2,
        )
        fast_selection = fast_policy.select_complete_stream(
            stream, task=task
        )
        reference_selection = (
            reference_policy.select_complete_stream(
                stream, task=task
            )
        )
        self.assertEqual(
            fast_selection.selected_candidate,
            reference_selection.selected_candidate,
        )
        self.assertEqual(
            fast_selection.earliest_candidate,
            reference_selection.earliest_candidate,
        )
        self.assertEqual(
            fast_selection.candidate_count,
            reference_selection.candidate_count,
        )
        self.assertEqual(
            fast_policy.pop_selection_traces()[0]
            .selected_candidate_features,
            reference_policy.pop_selection_traces()[0]
            .selected_candidate_features,
        )

    # DQN-009
    def test_estimate_plus_realization_correction_recovers_realized_utility(self):
        record = DecisionRecord("d", "t", "c", 0.0, 4.0)
        commit = RewardAssembler.commit_reward(record)
        correction = RewardAssembler.realization_correction(record, 3.0, 0.0)
        self.assertEqual(commit, 4.0)
        self.assertEqual(correction, -1.0)
        self.assertEqual(commit + correction, 3.0)

    # DQN-010 / DQN-011 / DQN-016
    def test_timestamped_system_events_enter_next_or_time_progression_transition(self):
        clock = GammaClock(0.99, TimeConverter(1.0))
        assembler = RewardAssembler(clock)
        assembler.buffer_event(TimestampedReward(5.0, 2.0, "prior", "EXPIRED"))
        transition = assembler.build_transition(
            global_state_before=(0.0,),
            selected_candidate_id=None,
            selected_candidate_features=None,
            immediate_reward=0.0,
            global_state_after=(1.0,),
            next_candidate_features=(),
            decision_time_sim=0.0,
            next_transition_time_sim=10.0,
            terminal=False,
        )
        self.assertAlmostEqual(transition.reward, 2.0 * (0.99 ** 5))
        self.assertAlmostEqual(transition.gamma_elapsed, 0.99 ** 10)
        self.assertEqual(transition.timestamped_event_rewards[0].event_type, "EXPIRED")

    # DQN-013 / DQN-014 / DQN-015 / DQN-017
    def test_discount_depends_only_on_physical_time_not_decision_count(self):
        clock = GammaClock(0.99, TimeConverter(1.0))
        self.assertEqual(clock.elapsed_seconds(2.0, 2.0), 0.0)
        self.assertEqual(clock.discount(0.0), 1.0)
        self.assertAlmostEqual(clock.discount(30.0), 0.99 ** 30)
        one_step = clock.discount(10.0)
        five_zero_steps_then_time = (clock.discount(0.0) ** 5) * clock.discount(10.0)
        self.assertEqual(one_step, five_zero_steps_then_time)

        aggregated = (clock.discount(5.0) * 2.0) + clock.discount(10.0) * 3.0
        explicit = clock.discount(5.0) * (2.0 + clock.discount(5.0) * 3.0)
        self.assertAlmostEqual(aggregated, explicit)

    # DQN-018
    def test_gamma_per_second_has_strict_domain(self):
        for invalid in (0.0, -1.0, 1.01, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(Exception):
                GammaClock(invalid, TimeConverter(1.0))
        unit = GammaClock(1.0, TimeConverter(300.0))
        self.assertEqual(unit.discount(999999.0), 1.0)

    def test_feature_schema_and_train_step_use_selected_candidate_features(self):
        config = CandidateFeatureConfig(10.0, 5.0, 1.0, 10.0, 100.0)
        encoder = CandidateFeatureEncoder({"N": 0}, config)
        self.assertEqual(encoder.feature_dim, 18)
        self.assertEqual(len(encoder.feature_schema_hash), 64)

        online = SharedCandidateQNetwork(2, 2, 4)
        target = SharedCandidateQNetwork(2, 2, 4)
        trainer = CandidateDQNTrainer(online, target, 1e-3)
        trainer.update_target()
        loss = trainer.train_transition(self._transition())
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
