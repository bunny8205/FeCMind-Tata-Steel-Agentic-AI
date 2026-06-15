"""Initialize data, models, and RAG index for the Maintenance Wizard project."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent import MaintenanceWizard  # noqa: E402


def main() -> None:
    wizard = MaintenanceWizard()
    wizard.initialize(force=True, load_llm=False)

    print("Initialization complete.")
    print("Assets:", wizard.asset_ids)
    print("RAG backend:", wizard.rag.backend)
    print("Asset health:")
    print(
        wizard.asset_health_table()[
            [
                "asset_id",
                "risk_band",
                "hybrid_failure_risk",
                "operational_rule_score",
                "estimated_rul_days",
            ]
        ].to_string(index=False)
    )

    result = wizard.chat("GBX-17 has abnormal vibration. Give root cause, priority, spares, and alert.")
    print("\nSmoke test answer preview:")
    print(result["answer"][:1200])


if __name__ == "__main__":
    main()
