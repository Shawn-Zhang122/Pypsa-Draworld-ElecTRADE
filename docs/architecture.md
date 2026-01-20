# Architecture Overview

## Dataflow

inputs → model → snapshot.json → GitHub → static web

## Layers

1. Model layer
   - PyPSA LP clearing
   - No UI, no API

2. Snapshot layer
   - JSON contract
   - Versioned via Git

3. Web layer
   - Read-only
   - Fetches latest.json

## Key principle
Model and UI never talk directly.
