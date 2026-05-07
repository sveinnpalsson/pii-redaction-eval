# Data and Model Licensing Notes

This repository separates code, synthetic benchmark data, third-party dataset inputs, model weights, and aggregate results.

## Code

The repository code, scripts, tests, and configuration files are released under
the MIT License in `LICENSE`.

## Synthetic Stress Fixture

The local-vault stress fixture was authored for this benchmark. It is synthetic and contains no real private user, customer, coworker, or operator data.

## AI4Privacy PII-Masking-300k

The AI4Privacy-derived validation and supporting fixtures are not redistributed here. Users must obtain the upstream dataset separately and comply with its license terms.

This repository provides reconstruction scripts, configuration files, label mappings, fixture hashes, and aggregate metrics rather than raw AI4Privacy-derived text or raw model outputs over that text.

The release metadata pins the upstream Hugging Face dataset revision used for
reconstruction commands.

## Model Weights

OpenAI Privacy Filter, Qwen, and other model weights are not redistributed here. Users must obtain model weights from their upstream sources and comply with the relevant model licenses.
