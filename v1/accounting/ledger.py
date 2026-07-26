"""Append-only, idempotent realized metrics ledger."""

from typing import Dict

from v1.domain.reservations import Reservation

from .energy import AccountingReport, ExogenousEnergyAccounting


class MetricsLedger:
    def __init__(self, accounting: ExogenousEnergyAccounting):
        self.accounting = accounting
        self._completed: Dict[str, Reservation] = {}
        self._finalized_report = None

    def record_completed_reservation(self, reservation: Reservation) -> bool:
        if self._finalized_report is not None:
            raise RuntimeError("cannot append after ledger finalization")
        existing = self._completed.get(reservation.reservation_id)
        if existing is not None:
            if existing != reservation:
                raise RuntimeError("reservation id collision in metrics ledger")
            return False
        self._completed[reservation.reservation_id] = reservation
        return True

    def finalize_after_full_settlement(self, accounting_interval=None) -> AccountingReport:
        if self._finalized_report is None:
            self._finalized_report = self.accounting.realize(
                self._completed.values(),
                accounting_interval=accounting_interval,
            )
        return self._finalized_report
