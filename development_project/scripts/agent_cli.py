"""Terminal chat for the Steel Plant Agent."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent import MaintenanceWizard  # noqa: E402


def main() -> None:
    wizard = MaintenanceWizard().initialize(load_llm=False)
    print("Steel Plant Agent ready. Type 'exit' to stop.")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        result = wizard.chat(query, user_id="cli_user")
        print("\nAgent:\n")
        print(result.get("answer", ""))


if __name__ == "__main__":
    main()
