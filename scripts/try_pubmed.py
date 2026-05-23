"""Manual smoke test for the PubMed tool against the real API.

Usage:
    python scripts/try_pubmed.py "sepsis[MeSH] AND 2024[PDAT]"
    python scripts/try_pubmed.py "qSOFA validation" --max-results 3

Set ``NCBI_API_KEY`` in your environment for the 10 req/sec limit.
"""

from __future__ import annotations

import argparse
import json
import sys

from clinical_agent.tools import PubMedSearchTool
from clinical_agent.utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="PubMed query string")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.verbose else "INFO")

    tool = PubMedSearchTool()
    result = tool.run(query=args.query, max_results=args.max_results)

    if not result.success:
        print(f"ERROR: {result.error}", file=sys.stderr)
        return 1

    for i, art in enumerate(result.data["articles"], 1):
        print(f"\n[{i}] PMID:{art['pmid']}  ({art.get('year') or 'n.d.'})")
        print(f"    {art['title']}")
        if art["authors"]:
            shown = ", ".join(art["authors"][:3])
            extra = f" +{len(art['authors']) - 3} more" if len(art["authors"]) > 3 else ""
            print(f"    {shown}{extra}")
        if art["journal"]:
            print(f"    Journal: {art['journal']}")
        if art["abstract"]:
            snippet = art["abstract"][:300]
            print(f"    Abstract: {snippet}{'...' if len(art['abstract']) > 300 else ''}")
        if art["mesh_terms"]:
            print(f"    MeSH: {', '.join(art['mesh_terms'][:5])}")

    print("\n---")
    print(json.dumps(result.metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
