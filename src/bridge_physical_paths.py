"""Portable CLI paths for the archived bridge recipe; never grants execution approval.

These settings are consumed before importing the separately supplied production
dependencies. Environment propagation supports spawned workers. This adapter
was not part of the historical numerical execution.
"""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

def consume_paths():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--production-project',default=os.environ.get('BRIDGE_PRODUCTION_PROJECT'))
    parser.add_argument('--bridge-root',default=os.environ.get('BRIDGE_RUN_ROOT'))
    parser.add_argument('--heavy-root',default=os.environ.get('BRIDGE_HEAVY_ROOT'))
    args,remaining=parser.parse_known_args()
    if not args.production_project or not args.bridge_root:
        parser.error('--production-project and --bridge-root (or their documented environment variables) are required')
    production=Path(args.production_project).expanduser().resolve()
    bridge=Path(args.bridge_root).expanduser().resolve()
    heavy=Path(args.heavy_root).expanduser().resolve() if args.heavy_root else None
    if not production.is_dir():parser.error('supplied production project does not exist')
    for output in (bridge,heavy):
        if output is not None and (output.is_relative_to(production) or production.is_relative_to(output)):
            parser.error('bridge output must be separate from production project, not an ancestor or child')
    os.environ['BRIDGE_PRODUCTION_PROJECT']=str(production)
    os.environ['BRIDGE_RUN_ROOT']=str(bridge)
    if heavy is not None:os.environ['BRIDGE_HEAVY_ROOT']=str(heavy)
    sys.argv=[sys.argv[0],*remaining]
    return SimpleNamespace(production_project=production,bridge_root=bridge,heavy_root=heavy)
