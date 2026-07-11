# Data provenance

## Authoritative metadata

- Dataset: [Crash data from Queensland roads](https://www.data.qld.gov.au/dataset/crash-data-from-queensland-roads)
- Resource: Road crash locations (`e88943c0-5968-4972-a15f-38e120d72ec0`)
- License: **Creative Commons Attribution 4.0**

## Working download

- Mirror: [Queensland OpenDataSoft](https://queensland.opendatasoft.com/explore/dataset/road-crash-locations-queensland/)
- Filter: `Brisbane City`, selected years and columns via API `where` / `select`
- Pre-seeded local file: `data/raw/brisbane_crashes_2015_2023.csv` (gitignored; checksum in config)

## Policy

1. OpenDataSoft export is a bandwidth-saving copy; official portal is metadata authority.
2. Acquisition saves URLs, timestamps, schema mapping, and row/byte counts to manifests.
3. Property-damage-only records excluded from severity modelling.
4. Years 2024+ excluded from principal benchmark unless marked complete in config.

## Attribution

© State of Queensland (Department of Transport and Main Roads).
This project is not affiliated with the Queensland Government.
